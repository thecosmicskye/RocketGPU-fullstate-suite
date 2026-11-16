"""Check RocketGPU's v3 layer (Car.get_state_v3) against RocketSim v3, one tick at a time, on the full-state suite.

    python score_v3_api.py DATA_DIR WORKER [--names a,b,...] [--splits dev,test,...] [--jobs N] [--json OUT]

WORKER is the v3/ build (target/release/fs_v3). Needs what score.py needs (PYTHONPATH with the build's python dir,
ROCKETGPU_COLLISION_MESH_DIR) and score.py's folder on PYTHONPATH when this file is not next to it.

Each segment is replayed in RocketGPU as score.py does. For every tick:
  prev  every car's get_state_v3(), the ball and the boost pads before the tick
  cur   every car's get_state_v3() after RocketGPU's step (before score.py's boost pad credits)
  ref   RocketSim v3 set to prev (each car's full CarState, the ball, the pads), given the same controls and stepped
        one tick: every car's CarState after it
and cur is compared with ref field by field. Bodies are not scored: the two simulators' physics differ. Each field is
counted over all compared car-ticks and over those where v3's is_on_ground and wheels_with_contact after the tick
equal RocketGPU's ("contacts agree").

What v3 gets and how values are mapped:
  - The v3 arena is kept for a segment. State v3 has no setter for (wheel suspension history, the world contact
    normal, pad membership, who demolished whom) is v3's own from its previous tick; the wheel contact gate is
    refreshed at the set pose (Arena::refresh_car_sticky_gate).
  - v3 has no driving-disabled state: a car RocketGPU holds (countdown, goal replay) gets the same controls in v3.
  - bump_cooldown_victim is a car id in RocketGPU and an arena index in v3: ids are mapped to indices.
  - last_extra_hit_tick counts each arena's own ticks (RocketGPU's is the full state's env_states.tick, which a
    restored segment starts at the game's tick; Arena.get_tick_count() restarts at 0): values are shifted by the
    difference of the two counts (a tick before v3's first is sent as its tick 0; the v3 arena is stepped 2 ticks
    first so that reads as old).
  - A pad RocketGPU has off is given its cooldown (at least one tick).

Rules for what is counted (each with its count in the output):
  - not compared: the ticks score.py does not score (a spawn segment's first tick, held ticks).
  - world_contact_normal is compared on car-ticks where both sides have one; has_world_contact covers the rest.
  - floats are equal when ==; "within 1e-5" is absolute.
Every mismatch among the contacts-agree car-ticks is put in one class: "demo" (the car is demolished before or after
the tick on either side), else "pad" (v3 or RocketGPU gave the car pad boost this tick), else "held" (driving
disabled), else "other".
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import subprocess
import time

import numpy as np

import score as S

PUB_F = 2 + 59        # a car's row in the data's "public": preset, team, then the v3 CarState floats
WARM = 2              # ticks the worker steps a new arena before the segment
NEAR = 1e-5
F32 = np.float32
CLASSES = ("demo", "pad", "held", "other")

# Columns of a car's row, RocketGPU's and the worker's alike: 18 body values, then the fields.
BODY = dict(pos=(0, 3), vel=(3, 6), ang_vel=(6, 9), rot_mat=(9, 18))
FIELDS = [   # name, first column, one past the last, is a float
    ("has_jumped", 18, 19, False), ("has_double_jumped", 19, 20, False), ("has_flipped", 20, 21, False),
    ("flip_rel_torque", 21, 24, True), ("jump_ticks", 24, 25, False), ("flip_time", 25, 26, True),
    ("is_flipping", 26, 27, False), ("is_jumping", 27, 28, False), ("jump_hold_broken", 28, 29, False),
    ("air_time", 29, 30, True), ("air_time_since_jump", 30, 31, True), ("boost", 31, 32, True),
    ("handbrake_val", 32, 33, True), ("is_supersonic", 33, 34, False), ("supersonic_grace_timer", 34, 35, True),
    ("time_since_boosted", 35, 36, True), ("boosting_time", 36, 37, True), ("is_boosting", 37, 38, False),
    ("is_auto_flipping", 38, 39, False), ("auto_flip_timer", 39, 40, True), ("auto_flip_torque_scale", 40, 41, True),
    ("bump_cooldown_timer", 41, 42, True), ("bump_cooldown_victim", 42, 43, False),
    ("last_extra_hit_tick", 43, 44, False), ("is_demoed", 44, 45, False), ("demo_respawn_timer", 45, 46, True),
    ("is_on_ground", 46, 47, False), ("wheels_with_contact", 47, 51, False), ("has_world_contact", 51, 52, False),
    ("world_contact_normal", 52, 55, True), ("controls", 55, 63, True), ("prev_controls", 63, 71, True),
]
COL = {name: a for name, a, _, _ in FIELDS}
NCOL, NREF = 71, 72   # the worker adds the pad a car picked up (-1 none)


def vec(v):
    return [v.x, v.y, v.z]


def ctl_row(c):
    return [c.throttle, c.steer, c.pitch, c.yaw, c.roll, float(c.jump), float(c.boost), float(c.handbrake)]


def car_row(v, slot_of):
    """A car's get_state_v3() as a row (the victim as an arena index, -3 for an unknown id; none is -1)."""
    m, w = v.rot_mat, v.world_contact_normal
    victim = v.bump_cooldown_victim
    return (vec(v.pos) + vec(v.vel) + vec(v.ang_vel) + vec(m.forward) + vec(m.right) + vec(m.up) +
            [v.has_jumped, v.has_double_jumped, v.has_flipped] + vec(v.flip_rel_torque) +
            [v.jump_ticks, v.flip_time, v.is_flipping, v.is_jumping, v.jump_hold_broken, v.air_time,
             v.air_time_since_jump, v.boost, v.handbrake_val, v.is_supersonic, v.supersonic_grace_timer,
             v.time_since_boosted, v.boosting_time, v.is_boosting, v.is_auto_flipping, v.auto_flip_timer,
             v.auto_flip_torque_scale, v.bump_cooldown_timer, -1 if victim is None else slot_of.get(int(victim), -3),
             -1 if v.last_extra_hit_tick is None else v.last_extra_hit_tick, v.is_demoed, v.demo_respawn_timer,
             v.is_on_ground] + list(v.wheels_with_contact) + [w is not None] + (vec(w) if w is not None else [0, 0, 0]) +
            ctl_row(v.controls) + ctl_row(v.prev_controls))


def read_cars(cars, slot_of):
    return np.array([car_row(c.get_state_v3(), slot_of) for c in cars], np.float64)


def replay(rp, rs, z, meta, n, at, credits):
    """Segment n in RocketGPU, as score.py's score() / controls_only_run() replay it. Per tick: prev and cur rows, the
    controls, which cars drive, the ball (18 + asleep), the pads' cooldowns, the arena's tick (env_states.tick) before the step
    and whether score.py scores the tick. Also the pads' positions."""
    seg, ncars, co = meta["segments"][n], meta["ncars"], meta.get("controls_only")
    arena, cars = S.restore(rp, rs, seg, meta["cars"][seg["cars"]],
                            {k[2:]: z[k][n].tobytes() for k in z if k.startswith("s.")})
    slot_of = {c.id: j for j, c in enumerate(cars)}
    pad_xy = [None] * len(arena.get_boost_pad_states())
    for p in arena.get_boost_pads():
        pos = p.get_pos()
        pad_xy[p.pad_state_index] = (pos.x, pos.y)
    T = seg["ticks"]
    out = dict(prev=np.zeros((T, ncars, NCOL)), cur=np.zeros((T, ncars, NCOL)), ctl=np.zeros((T, ncars, 8)),
               driving=np.zeros((T, ncars), bool), ball=np.zeros((T, 19)), pads=np.zeros((T, len(pad_xy))),
               tick=np.zeros(T, np.int64), scored=np.zeros(T, bool), pad_xy=np.array(pad_xy, np.float64),
               teams=[int(c.team) for c in cars])
    cur = None
    for k in range(T):
        i = at[seg["start"] + k]
        for j, c in enumerate(cars):
            out["ctl"][k, j] = z["ctl"][i, seg["perm"][j] if co else j]
            c.set_controls(S.controls(rs, out["ctl"][k, j]))
        if not co:
            if not seg["timer_hold"]:
                S.set_driving(arena, [int(d) for d in z["drv"][i]])
            S.set_frame(arena, None if np.isnan(z["frac"][i]) else float(z["frac"][i]),
                        float(z["start_dt"][i]) if k else -1.0)
        st = S.State(arena)
        fmt = S.INT_FMT[st.span("car_states.driving_disabled", 0)[1]]
        out["driving"][k] = [not st.get("car_states.driving_disabled", j, fmt)[0] for j in range(ncars)]
        b = arena.ball.get_state()
        asleep = st.get("ball_states.sleeping", 0, S.INT_FMT[st.span("ball_states.sleeping", 0)[1]])[0]
        out["ball"][k] = (vec(b.pos) + vec(b.rot_mat.forward) + vec(b.rot_mat.right) + vec(b.rot_mat.up) + vec(b.vel) +
                          vec(b.ang_vel) + [bool(asleep)])
        out["pads"][k] = [0.0 if p.is_active else max(p.cooldown, 1.0 / 120.0) for p in arena.get_boost_pad_states()]
        out["tick"][k] = int.from_bytes(arena.get_full_state(["env_states"])["env_states.tick"], "little")
        out["prev"][k] = read_cars(cars, slot_of) if cur is None else cur
        arena.step(1)
        cur = out["cur"][k] = read_cars(cars, slot_of)
        if not co and i in credits:
            S.credit_boost(arena, credits[i])
            cur = None
        out["scored"][k] = not (seg["spawn"] and k == 0 or z["held"][i])
    return out


def v3_step(w, run, presets):
    """Every tick of a replayed segment through the worker: v3's rows after each one-tick step from prev."""
    T, ncars = run["prev"].shape[:2]
    prev = run["prev"]
    D = run["tick"] - (WARM + np.arange(T))          # RocketGPU's tick count minus v3's
    hit = prev[:, :, COL["last_extra_hit_tick"]]
    sent_hit = np.where(hit < 0, -1.0, np.maximum(hit - D[:, None], 0.0))
    state = prev[:, :, 18:51].copy()
    state[:, :, COL["last_extra_hit_tick"] - 18] = sent_hit
    state[:, :, COL["bump_cooldown_victim"] - 18] = np.maximum(state[:, :, COL["bump_cooldown_victim"] - 18], -1.0)
    # the worker's car row: pos, forward, right, up, vel, ang vel, this tick's controls, the state, then the
    # previous controls
    cars = np.concatenate([prev[:, :, 0:3], prev[:, :, 9:18], prev[:, :, 3:9], run["ctl"], state,
                           prev[:, :, COL["prev_controls"]:NCOL]], axis=2)
    head = [-2.0, ncars, T]
    for j in range(ncars):
        head += [presets[j], run["teams"][j]]
    head += [len(run["pad_xy"])] + list(run["pad_xy"].reshape(-1))
    body = np.concatenate([cars.reshape(T, -1), run["ball"], run["pads"]], axis=1)
    w.stdin.write(np.array(head, "<f8").tobytes() + body.astype("<f8").tobytes())
    w.stdin.flush()
    size = 4 * T * (ncars * NREF + 18)
    raw = w.stdout.read(size)
    if len(raw) != size:
        raise RuntimeError("the v3 worker stopped")
    ref = np.frombuffer(raw, "<f4").reshape(T, ncars * NREF + 18)[:, :ncars * NREF].reshape(T, ncars, NREF).copy()
    # v3's hit tick back in RocketGPU's count; one it did not change is prev's own
    r = ref[:, :, COL["last_extra_hit_tick"]].astype(np.float64)
    ref_hit = np.where(r == sent_hit, hit, np.where(r < 0, -1.0, r + D[:, None]))
    return ref, ref_hit


def fmt_val(x, is_float):
    return "(" + ",".join(("%.9g" % v) if is_float else str(int(v)) for v in x) + ")" if len(x) > 1 else \
        ("%.9g" % x[0]) if is_float else str(int(x[0]))


def new_result(name):
    return dict(name=name, segments=0, car_ticks=0, agree=0, not_compared=0, classes=dict.fromkeys(CLASSES, 0),
                classes_agree=dict.fromkeys(CLASSES, 0), unknown_victim_id=0, v3_pickups=0, our_boost_gains=0,
                fields={f[0]: dict(all=[0, 0, 0], agree=[0, 0, 0], cls={}) for f in FIELDS},
                bodies={b: [0, 0] for b in BODY})


def tally(r, seg, run, ref, ref_hit):
    """Add one segment's comparison to the recording's result."""
    prev, cur = run["prev"].astype(F32), run["cur"].astype(F32)
    scored = np.broadcast_to(run["scored"][:, None], cur.shape[:2])
    r["segments"] += 1
    r["not_compared"] += int((~scored).sum())
    r["car_ticks"] += int(scored.sum())
    r["unknown_victim_id"] += int((cur[:, :, COL["bump_cooldown_victim"]][scored] == -3).sum())
    refc = ref[:, :, :NCOL].copy()
    refc[:, :, COL["last_extra_hit_tick"]] = ref_hit
    dem = COL["is_demoed"]
    demo = (prev[:, :, dem] != 0) | (cur[:, :, dem] != 0) | (ref[:, :, dem] != 0)
    v3_pick = ref[:, :, NCOL] != -1
    our_gain = (cur[:, :, COL["boost"]] > prev[:, :, COL["boost"]]) & (cur[:, :, dem] == prev[:, :, dem])
    r["v3_pickups"] += int((v3_pick & scored).sum())
    r["our_boost_gains"] += int((our_gain & scored).sum())
    cls = np.where(demo, 0, np.where(v3_pick | our_gain, 1, np.where(~run["driving"], 2, 3)))
    g, w = COL["is_on_ground"], COL["wheels_with_contact"]
    agree = scored & (cur[:, :, g] == ref[:, :, g]) & (cur[:, :, w:w + 4] == ref[:, :, w:w + 4]).all(axis=2)
    r["agree"] += int(agree.sum())
    for c, name in enumerate(CLASSES):
        r["classes"][name] += int((scored & (cls == c)).sum())
        r["classes_agree"][name] += int((agree & (cls == c)).sum())
    for name, (a, b) in BODY.items():
        r["bodies"][name][0] += int(((cur[:, :, a:b] == ref[:, :, a:b]).all(axis=2) & scored).sum())
        r["bodies"][name][1] += int(scored.sum())
    both = (cur[:, :, COL["has_world_contact"]] != 0) & (ref[:, :, COL["has_world_contact"]] != 0)
    for name, a, b, is_float in FIELDS:
        o, v = cur[:, :, a:b], refc[:, :, a:b]
        eq = (o == v).all(axis=2)
        near = (np.abs(o.astype(np.float64) - v) <= NEAR).all(axis=2) if is_float else eq
        base = both if name == "world_contact_normal" else True
        f = r["fields"][name]
        for key, pop in (("all", scored & base), ("agree", agree & base)):
            f[key][0] += int((eq & pop).sum())
            f[key][1] += int((near & pop).sum())
            f[key][2] += int(pop.sum())
        bad = agree & base & ~eq
        if bad.any():
            for c, cname in enumerate(CLASSES):
                idx = np.argwhere(bad & (cls == c))
                if not len(idx):
                    continue
                d = f["cls"].setdefault(cname, dict(n=0, pairs={}, first=[]))
                d["n"] += len(idx)
                rows, counts = np.unique(np.concatenate([o[idx[:, 0], idx[:, 1]], v[idx[:, 0], idx[:, 1]]], axis=1),
                                         axis=0, return_counts=True)
                for row, cnt in zip(rows, counts):
                    key = fmt_val(row[:b - a], is_float) + " vs " + fmt_val(row[b - a:], is_float)
                    d["pairs"][key] = d["pairs"].get(key, 0) + int(cnt)
                for k, j in idx[:3 - len(d["first"])]:
                    d["first"].append([r["name"], seg["start"], int(k), int(j)])


def trim(r, keep=40):
    for f in r["fields"].values():
        for d in f["cls"].values():
            d["pairs"] = dict(collections.Counter(d["pairs"]).most_common(keep))
    return r


def run(args):
    path, worker = args
    rp, rs = S.build()
    z = np.load(path)
    z = {k: z[k] for k in z.files}
    meta = json.loads(z["meta"].tobytes())
    at = {int(f): i for i, f in enumerate(z["frame"])}
    credits = {}
    if not meta.get("controls_only"):
        for j, s, g in zip(z["credit_tick"], z["credit_slot"], z["credit_gain"]):
            credits.setdefault(int(j), []).append((int(s), float(g)))
    w = subprocess.Popen([worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    r = new_result(meta["name"])
    for n, seg in enumerate(meta["segments"]):
        rep = replay(rp, rs, z, meta, n, at, credits)
        pub = z["public"][n]
        teams = [int(pub[j * PUB_F + 1]) for j in range(meta["ncars"])]
        if teams != rep["teams"]:
            raise RuntimeError(f"{meta['name']} segment {n}: teams {rep['teams']} != the public row's {teams}")
        ref, ref_hit = v3_step(w, rep, [pub[j * PUB_F] for j in range(meta["ncars"])])
        tally(r, seg, rep, ref, ref_hit)
    w.stdin.write(np.array([-1.0], "<f8").tobytes())
    w.stdin.close()
    w.wait()
    return trim(r)


def merge(res):
    t = new_result("total")
    for r in res:
        for k in ("segments", "car_ticks", "agree", "not_compared", "unknown_victim_id", "v3_pickups", "our_boost_gains"):
            t[k] += r[k]
        for k in ("classes", "classes_agree"):
            for c in CLASSES:
                t[k][c] += r[k][c]
        for b in BODY:
            t["bodies"][b] = [p + q for p, q in zip(t["bodies"][b], r["bodies"][b])]
        for name, f in r["fields"].items():
            g = t["fields"][name]
            for key in ("all", "agree"):
                g[key] = [p + q for p, q in zip(g[key], f[key])]
            for c, d in f["cls"].items():
                e = g["cls"].setdefault(c, dict(n=0, pairs={}, first=[]))
                e["n"] += d["n"]
                for p, cnt in d["pairs"].items():
                    e["pairs"][p] = e["pairs"].get(p, 0) + cnt
                e["first"] += d["first"][:3 - len(e["first"])]
    return t


def report(r, details):
    def cell(x):
        eq, near, n = x
        pct = "%8.4f%%" % (100.0 * eq / n) if n else "     n/a "
        return f"{eq:>10} / {n:<10} {pct}"

    print(f"== {r['name']}: {r['segments']} segments, {r['car_ticks']} car-ticks compared "
          f"({r['not_compared']} not compared: ticks score.py does not score), contacts agree on {r['agree']}")
    print("   car-ticks by class, all / contacts agree: " +
          ", ".join(f"{c} {r['classes'][c]} / {r['classes_agree'][c]}" for c in CLASSES) +
          f"; v3 pad pickups {r['v3_pickups']}, RocketGPU boost gains in the step {r['our_boost_gains']}, "
          f"victim ids not in the arena {r['unknown_victim_id']}")
    print(f"   {'field':<24}{'all: equal / compared':^36}{'within 1e-5':>12}   {'contacts agree: equal / compared':^36}"
          f"{'within 1e-5':>12}")
    for name, _, _, is_float in FIELDS:
        f = r["fields"][name]
        print(f"   {name:<24}{cell(f['all'])}{f['all'][1] if is_float else '':>12}   {cell(f['agree'])}"
              f"{f['agree'][1] if is_float else '':>12}")
    print("   bodies equal (not scored): " + ", ".join(f"{b} {e} / {n}" for b, (e, n) in r["bodies"].items()))
    if not details:
        return
    print("   mismatches among contacts-agree car-ticks, by class: count, commonest RocketGPU vs v3 values, first "
          "places (recording, segment start frame, tick, car slot)")
    for name, _, _, _ in FIELDS:
        f = r["fields"][name]
        for c in CLASSES:
            d = f["cls"].get(c)
            if d:
                pairs = ", ".join(f"{p} x{n}" for p, n in collections.Counter(d["pairs"]).most_common(4))
                print(f"   {name} [{c}] {d['n']}: {pairs}; first {d['first']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("worker")
    ap.add_argument("--names", help="comma-separated recordings (default: every recording of the splits)")
    ap.add_argument("--splits", default="all", help="comma-separated splits from manifest.json, or all")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args()
    man = json.load(open(os.path.join(S.HERE, "manifest.json")))["recordings"]
    splits = [x["split"] for x in man] if a.splits == "all" else a.splits.split(",")
    names = a.names.split(",") if a.names else [x["name"] for x in man if x["split"] in splits]
    worker = os.path.abspath(a.worker)
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.jobs, len(names))) as pool:
        res = pool.map(run, [(os.path.join(a.data, n + ".npz"), worker) for n in names], chunksize=1)
    print(f"{len(names)} recordings, {time.time() - t0:.0f} s")
    total = merge(res)
    if len(res) > 1:
        for r in res:
            report(r, False)
    report(total, True)
    if a.json:
        json.dump(dict(total=total, recordings=res), open(a.json, "w"))


if __name__ == "__main__":
    main()
