"""Score a RocketGPU build on the full-state suite.

    python score.py DATA_DIR [--splits dev,test,...] [--jobs N] [--shard I/N] [--json OUT]
    python score.py --merge OUT.json PART.json...

Needs PYTHONPATH with the build's python dir (rocketgpu_py and rocketsim) and ROCKETGPU_COLLISION_MESH_DIR. The start
states go in through Arena.set_full_state.

Each segment starts from the game's full state at a tick and runs open loop for up to 120 ticks on the game's
recorded inputs. Every tick is compared with the game's state:
  exact          body-ticks (each car and the ball) whose position, velocity, angular velocity and rotation all match
                 the game to the bit, and segments whose every body-tick does.
  segment pass   car-ticks whose normalized error (pos / 2 uu, vel / 3 uu/s, ang vel / 0.05 rad/s, forward and up
                 axes / 0.05, over the car and the ball) is below 1.
  mean error     that normalized error, capped at 10, at 1, 30 and 120 ticks.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import struct
import time

import numpy as np

UU = 50.0
TOL = dict(pos=2.0, vel=3.0, ang=0.05, axis=0.05)
KS = (1, 30, 120)
HERE = os.path.dirname(os.path.abspath(__file__))


# ---- the build ----

def build():
    import rocketgpu_py as rp
    import rocketsim as rs
    return rp, rs


class State:
    """The arena's car and ball fields (Arena.get_full_state), read and written back by name."""

    def __init__(self, arena):
        self.arena, self.st, self.new = arena, arena.get_full_state(["car_states", "ball_states"]), {}
        self.cars = len(self.st["car_states.boost"]) // 4

    def span(self, key, k):
        n = len(self.st[key]) // (self.cars if key.startswith("car_states.") else 1)
        return k * n, n

    def get(self, key, k, fmt):
        return struct.unpack_from(fmt, self.new.get(key, self.st[key]), self.span(key, k)[0])

    def put(self, key, k, fmt, *v):
        struct.pack_into(fmt, self.new.setdefault(key, bytearray(self.st[key])), self.span(key, k)[0], *v)

    def commit(self):
        if self.new:
            self.arena.set_full_state({k: bytes(v) for k, v in self.new.items()})


INT_FMT = {1: "<B", 2: "<H", 4: "<I"}


def from_obj(mods, o):
    """Rebuild a value the export wrote with to_obj (pybind objects by type name, enums by value)."""
    if isinstance(o, list):
        return [from_obj(mods, x) for x in o]
    if not isinstance(o, dict):
        return o
    if "__enum__" in o:
        cls = next(getattr(m, o["__enum__"]) for m in mods if hasattr(m, o["__enum__"]))
        return next(v for v in cls.__members__.values() if int(v) == o["value"])
    if "__type__" in o:
        x = next(getattr(m, o["__type__"]) for m in mods if hasattr(m, o["__type__"]))()
        for k, v in o.items():
            if k != "__type__":
                setattr(x, k, from_obj(mods, v))
        return x
    return {k: from_obj(mods, v) for k, v in o.items()}


def restore(rp, rs, seg, cars_log, state):
    """A fresh arena holding the segment's start: the cars, the boost pad and visit order calls, the mutators, then
    the full state."""
    mods = (rs, rp.compat, rp)
    cfg = rp.compat.ArenaConfig()
    cfg.no_ball_rot = False
    arena = rs.Arena(rs.GameMode.SOCCAR, cfg, 120.0)
    cars = [arena.add_car(*from_obj(mods, args), **from_obj(mods, kw)) for _, args, kw in cars_log]
    if [c.id for c in cars] != seg["ids"]:
        raise RuntimeError(f"car ids {[c.id for c in cars]} != {seg['ids']}")
    for name, args, kw in seg["calls"]:
        getattr(arena, name)(*from_obj(mods, args), **from_obj(mods, kw))
    m = arena.get_mutator_config()
    for k, v in seg["mutator"].items():
        if k != "__type__":
            setattr(m, k, from_obj(mods, v))
    arena.set_mutator_config(m)
    if state:
        try:
            arena.set_full_state(state)
        except ValueError as e:
            raise RuntimeError(f"this RocketGPU build doesn't match the suite's fields: {e}") from None
    return arena, cars


def controls(rs, c):
    out = rs.CarControls()
    out.throttle, out.steer, out.pitch, out.yaw, out.roll = (float(x) for x in c[:5])
    out.jump, out.boost, out.handbrake = bool(c[5]), bool(c[6]), bool(c[7])
    return out


def set_driving(arena, drv):
    """Cars the game holds this tick (kickoff countdown, goal replay): driving_disabled, and the ball's 2D constraint
    while every car is held."""
    st = State(arena)
    fmt = INT_FMT[st.span("car_states.driving_disabled", 0)[1]]
    for slot, d in enumerate(drv):
        st.put("car_states.driving_disabled", slot, fmt, 0 if d else 1)
    fmt = INT_FMT[st.span("ball_states.constrained_2d", 0)[1]]
    want = 1 if len(drv) and not any(drv) else 0
    if st.get("ball_states.constrained_2d", 0, fmt)[0] != want:
        st.put("ball_states.constrained_2d", 0, fmt, want)
    st.commit()


def set_frame(arena, frac, start_dt):
    """The game's render frame for this tick: its interpolation fraction (boost pad test; -1 on a frame's earlier
    ticks, None when not recorded) and, on a frame's first tick, its time for the countdown (-1 otherwise)."""
    m = arena.get_mutator_config()
    dt = m.frame_start_dt != start_dt
    if dt:
        m.frame_start_dt = float(start_dt)
    if frac is not None:
        m.boost_pad_frame_frac = float(frac)
    if dt or frac is not None:
        arena.set_mutator_config(m)


def credit_boost(arena, credits):
    """The game's boost pad credits after a tick: (slot among live cars, gain 0..1); a small pad adds 12, a big one
    fills to 100."""
    st = State(arena)
    n = st.cars
    live = [j for j in range(n) if not st.get("car_states.is_demoed", j, "<B")[0]]
    for slot, gain in credits:
        if len(live) < n and slot < len(live):
            slot = live[slot]
        b = st.get("car_states.boost", slot, "<f")[0]
        m = round(gain / 0.12)
        st.put("car_states.boost", slot, "<f",
               min(100.0, b + 12.0 * m) if 1 <= m <= 4 and abs(gain - 0.12 * m) < 1e-3 else 100.0)
    st.commit()


def read_bodies(arena, n):
    """Each car and the ball after a step, in Bullet units (bt = uu / 50); basis vectors are columns."""
    st = State(arena)
    out = []
    for k in range(n):
        g = lambda f: st.get("car_states." + f, k, "<3f")
        vel_uu = g("internal_vel")
        out.append(dict(pos=g("pos_bt"), vel=tuple(x / UU for x in vel_uu), vel_bt=g("vel_bt"), vel_uu=vel_uu,
                        ang=g("internal_ang_vel"), fwd=g("basis_forward"), right=g("basis_right"), up=g("basis_up")))
    g = lambda f: st.get("ball_states." + f, 0, "<3f")
    vel_uu = g("vel")
    out.append(dict(pos=g("pos_bt"), vel=tuple(x / UU for x in vel_uu), vel_bt=g("vel_bt"), vel_uu=vel_uu,
                    ang=g("ang_vel"), fwd=g("basis_forward"), right=g("basis_right"), up=g("basis_up")))
    return out


# ---- metrics ----

def exact(s, t):
    """pos, vel, ang vel and the 9 basis entries bit for bit against the game's (truth rows: pos, vel, ang vel in bt,
    then the basis by rows)."""
    basis = np.array(list(s["fwd"]) + list(s["right"]) + list(s["up"]), np.float32).reshape(3, 3).T.reshape(9)
    x = np.concatenate([np.asarray(s["pos"], np.float32), np.asarray(s["vel_bt"], np.float32),
                        np.asarray(s["ang"], np.float32), basis])
    return x.tobytes() == np.asarray(t[:18], np.float32).tobytes()


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def truth_rows(tr):
    out = []
    for b in tr:
        r = [b[9:12], b[12:15], b[15:18]]
        col = lambda c: tuple(r[i][c] for i in range(3))
        out.append(dict(pos=b[0:3], vel=b[3:6], ang=b[6:9], fwd=col(0), up=col(2)))
    return out


def norms(sim, tru, ncars):
    def terms(s, t):
        return [dist(s["pos"], t["pos"]) * UU / TOL["pos"], dist(s["vel"], t["vel"]) * UU / TOL["vel"],
                dist(s["ang"], t["ang"]) / TOL["ang"], dist(s["fwd"], t["fwd"]) / TOL["axis"],
                dist(s["up"], t["up"]) / TOL["axis"]]
    ball = terms(sim[ncars], tru[ncars])
    return [math.sqrt(sum(v * v for v in terms(sim[k], tru[k]) + ball)) for k in range(ncars)]


# ---- scoring ----

def score(path):
    rp, rs = build()
    z = np.load(path)
    meta = json.loads(z["meta"].tobytes())
    ncars = meta["ncars"]
    state = {k[2:]: z[k] for k in z.files if k.startswith("s.")}
    ctl, drv, frac, start_dt = z["ctl"], z["drv"], z["frac"], z["start_dt"]
    held, truth = z["held"], z["truth"]
    at = {int(f): i for i, f in enumerate(z["frame"])}
    credits = {}
    for j, s, g in zip(z["credit_tick"], z["credit_slot"], z["credit_gain"]):
        credits.setdefault(int(j), []).append((int(s), float(g)))
    r = dict(name=meta["name"], passed=0, total=0, at_k={str(k): [] for k in KS}, exact_bt=0, body_ticks=0, exact_segs=0,
             segs=0, first=[])
    for n, seg in enumerate(meta["segments"]):
        arena, cars = restore(rp, rs, seg, meta["cars"][seg["cars"]], {k: v[n].tobytes() for k, v in state.items()})
        seg_exact, scored = True, False
        for k in range(seg["ticks"]):
            i = at[seg["start"] + k]
            for slot, c in enumerate(cars):
                c.set_controls(controls(rs, ctl[i, slot]))
            if not seg["timer_hold"]:
                set_driving(arena, [int(d) for d in drv[i]])
            set_frame(arena, None if np.isnan(frac[i]) else float(frac[i]), float(start_dt[i]) if k else -1.0)
            arena.step(1)
            if i in credits:
                credit_boost(arena, credits[i])
            if seg["spawn"] and k == 0 or held[i]:
                continue
            sim = read_bodies(arena, ncars)
            for e in norms(sim, truth_rows(truth[i]), ncars):
                r["total"] += 1
                r["passed"] += e < 1.0
                if str(k + 1) in r["at_k"]:
                    r["at_k"][str(k + 1)].append(min(e, 10.0))
            scored = True
            for j in range(ncars + 1):
                ok = exact(sim[j], truth[i][j])
                r["body_ticks"] += 1
                r["exact_bt"] += ok
                if not ok and seg_exact:
                    seg_exact = False
                    if len(r["first"]) < 5:
                        r["first"].append([seg["start"], k, "ball" if j == ncars else f"car{j}"])
        if scored:
            r["segs"] += 1
            r["exact_segs"] += seg_exact
    return r


def summary(res, groups):
    out = {}
    for grp, names in groups.items():
        rs_ = [r for r in res if r["name"] in names]
        if not rs_:
            continue
        P, N = sum(r["passed"] for r in rs_), sum(r["total"] for r in rs_)
        EB, BT = sum(r["exact_bt"] for r in rs_), sum(r["body_ticks"] for r in rs_)
        ES, SG = sum(r["exact_segs"] for r in rs_), sum(r["segs"] for r in rs_)
        mk = {k: float(np.mean([x for r in rs_ for x in r["at_k"][str(k)]] or [0.0])) for k in KS}
        out[grp] = dict(recordings=len(rs_), exact_body_ticks=100.0 * EB / max(BT, 1), body_ticks=BT,
                        exact_segments=100.0 * ES / max(SG, 1), segments=SG, segment_pass=100.0 * P / max(N, 1),
                        car_ticks=N, mean_err=mk)
    return out


def report(out, res):
    for grp, d in out.items():
        print(f"[{grp}] {d['recordings']} recordings  exact body-ticks {d['exact_body_ticks']:.4f}% of {d['body_ticks']}"
              f"  segments {d['exact_segments']:.3f}% of {d['segments']}  segment pass {d['segment_pass']:.3f}% of "
              f"{d['car_ticks']} car-ticks  mean err " + "  ".join(f"k{k} {x:.4f}" for k, x in d["mean_err"].items()))
    bad = [r for r in res if r["exact_bt"] < r["body_ticks"]]
    for r in bad:
        print(f"  {r['name']}: exact {100.0 * r['exact_bt'] / r['body_ticks']:.4f}%  first inexact (start, tick, body) "
              f"{r['first']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?")
    ap.add_argument("--splits", default="all", help="comma-separated splits from manifest.json, or all")
    ap.add_argument("--jobs", type=int, default=4, help="worker processes (each holds its own arena)")
    ap.add_argument("--shard", default="0/1", help="I/N: score every Nth recording from the I-th (merge with --merge)")
    ap.add_argument("--json")
    ap.add_argument("--merge", nargs="+", metavar=("OUT", "PART"))
    a = ap.parse_args()
    man = json.load(open(os.path.join(HERE, "manifest.json")))["recordings"]
    split_of = {r["name"]: r["split"] for r in man}
    if a.merge:
        res = [r for p in a.merge[1:] for r in json.load(open(p))["recordings"]]
        splits = sorted({split_of[r["name"]] for r in res}, key=[r["split"] for r in man].index)
    else:
        splits = [r["split"] for r in man] if a.splits == "all" else a.splits.split(",")
        splits = list(dict.fromkeys(splits))
        si, sn = map(int, a.shard.split("/"))
        names = [r["name"] for r in man if r["split"] in splits][si::sn]
        t0 = time.time()
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            res = pool.map(score, [os.path.join(a.data, n + ".npz") for n in names], chunksize=1)
        print(f"{len(names)} recordings, {time.time() - t0:.0f} s")
    groups = {s: {r["name"] for r in res if split_of[r["name"]] == s} for s in splits}
    groups["all"] = {r["name"] for r in res}
    out = summary(res, groups)
    report(out, res)
    path = a.merge[0] if a.merge else a.json
    if path:
        json.dump(dict(summary=out, recordings=res), open(path, "w"))


if __name__ == "__main__":
    main()
