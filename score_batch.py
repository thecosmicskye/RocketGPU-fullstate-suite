"""score.py with a recording's segments stepped together as the envs of one DevBatch.

    python score_batch.py DATA_DIR [score.py args]

Each segment is restored as score.py restores it (on an arena reused per car list: pads, calls, mutator and full
state written fresh), copied into its env, and stepped with the same per-tick inputs, holds, frame fracs, credits and
reads. Spawn segments draw on respawn state the copy does not carry, so they run on score.py's own path. Results are
added in segment order, so the JSON matches score.py's.
"""
import json
import os
import sys

import numpy as np

import score as S

BATCH_MAX = 256
W = (np.float32(1) - 1.0).dtype   # what score.py's float - float32 scalar computes in
CTL = np.dtype([("f", "<f4", 5), ("b", "u1", 3), ("pad", "u1")])
ARENAS = {}


def arena_for(rp, rs, cars_log):
    key = json.dumps(cars_log, sort_keys=True)
    if key not in ARENAS:
        mods = (rs, rp.compat, rp)
        cfg = rp.compat.ArenaConfig()
        cfg.no_ball_rot = False
        arena = rs.Arena(rs.GameMode.SOCCAR, cfg, 120.0)
        cars = [arena.add_car(*S.from_obj(mods, a), **S.from_obj(mods, kw)) for _, a, kw in cars_log]
        ARENAS[key] = (arena, cars, arena.get_boost_pad_states())
    return ARENAS[key]


def restore_into(rp, rs, seg, cars_log, state):
    arena, cars, pads = arena_for(rp, rs, cars_log)
    mods = (rs, rp.compat, rp)
    if [c.id for c in cars] != seg["ids"]:
        raise RuntimeError(f"car ids {[c.id for c in cars]} != {seg['ids']}")
    arena.set_boost_pad_states(pads)
    for name, args, kw in seg["calls"]:
        getattr(arena, name)(*S.from_obj(mods, args), **S.from_obj(mods, kw))
    m = arena.get_mutator_config()
    for k, v in seg["mutator"].items():
        if k != "__type__":
            setattr(m, k, S.from_obj(mods, v))
    arena.set_mutator_config(m)
    arena.set_full_state(state)
    return arena, cars


def py_sum(xs):
    """Python's sum() of floats, elementwise: from 3.12 Neumaier-compensated."""
    f = xs[0] + 0.0
    if sys.version_info < (3, 12):
        for x in xs[1:]:
            f = f + x
        return f
    c = np.zeros_like(f)
    for x in xs[1:]:
        t = f + x
        c = c + np.where(np.abs(f) >= np.abs(x), (f - t) + x, (x - t) + f)
        f = t
    return np.where((c != 0) & np.isfinite(c), f + c, f)


class Fields:
    """Byte spans of the fields score.py reads and writes, in one car / ball element."""

    def __init__(self, rp):
        d = rp.compat.Arena.dev_layouts()
        self.car, self.ball = d["CarState"][1], d["BallState"][1]
        self.c = {k[9:]: v for k, v in d.items() if k.startswith("CarState.")}
        self.b = {k[10:]: v for k, v in d.items() if k.startswith("BallState.")}


def score(path):
    rp, rs = S.build()
    F = Fields(rp)
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
    segs = meta["segments"]
    out = [None] * len(segs)   # per segment: (passed, total, at_k, body_ticks, exact_bt, scored, seg_exact, first)

    def tally(n, sim, i, k, acc):
        for e in S.norms(sim, S.truth_rows(truth[i]), ncars):
            acc[1] += 1
            acc[0] += e < 1.0
            if str(k + 1) in acc[2]:
                acc[2][str(k + 1)].append(min(e, 10.0))
        acc[5] = True
        for j in range(ncars + 1):
            ok = S.exact(sim[j], truth[i][j])
            acc[3] += 1
            acc[4] += ok
            if not ok and acc[6]:
                acc[6] = False
                acc[7] = [segs[n]["start"], k, "ball" if j == ncars else f"car{j}"]

    def new_acc():
        return [0, 0, {str(k): [] for k in S.KS}, 0, 0, False, True, None]

    def single(n):
        seg = segs[n]
        arena, cars = S.restore(rp, rs, seg, meta["cars"][seg["cars"]], {k: v[n].tobytes() for k, v in state.items()})
        acc = new_acc()
        for k in range(seg["ticks"]):
            i = at[seg["start"] + k]
            for slot, c in enumerate(cars):
                c.set_controls(S.controls(rs, ctl[i, slot]))
            if not seg["timer_hold"]:
                S.set_driving(arena, [int(d) for d in drv[i]])
            S.set_frame(arena, None if np.isnan(frac[i]) else float(frac[i]), float(start_dt[i]) if k else -1.0)
            arena.step(1)
            if i in credits:
                S.credit_boost(arena, credits[i])
            if seg["spawn"] and k == 0 or held[i]:
                continue
            tally(n, S.read_bodies(arena, ncars), i, k, acc)
        out[n] = acc

    def run(batch):
        b, cur_frac = None, []
        for e, n in enumerate(batch):
            seg = segs[n]
            arena, cars = restore_into(rp, rs, seg, meta["cars"][seg["cars"]],
                                       {k: v[n].tobytes() for k, v in state.items()})
            if b is None:
                b = rp.compat.DevBatch(arena, len(batch))
            b.set_env(e, arena)
            cur_frac.append(arena.get_mutator_config().boost_pad_frame_frac)
        b.commit()
        cap, N = b.car_capacity, len(batch)
        accs = [new_acc() for _ in batch]
        cs = np.frombuffer(bytearray(b.get_raw("car_states")), np.uint8).reshape(N * cap, F.car).copy()
        bs = np.frombuffer(bytearray(b.get_raw("ball_states")), np.uint8).reshape(N, F.ball).copy()
        dd_o, dd_n = F.c["driving_disabled"]
        c2_o, c2_n = F.b["constrained_2d"]
        bo_o, _ = F.c["boost"]
        dm_o, _ = F.c["is_demoed"]
        cb = np.zeros(N * cap, CTL)
        for k in range(max(segs[n]["ticks"] for n in batch)):
            live = [e for e, n in enumerate(batch) if k < segs[n]["ticks"]]
            cs_dirty = bs_dirty = False
            dts = [-1.0] * N
            for e in live:
                seg = segs[batch[e]]
                i = at[seg["start"] + k]
                for slot in range(len(meta["cars"][seg["cars"]])):
                    c = ctl[i, slot]
                    cb[e * cap + slot] = ((float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4])),
                                          (bool(c[5]), bool(c[6]), bool(c[7])), 0)
                if not seg["timer_hold"]:
                    d = [int(x) for x in drv[i]]
                    for slot, x in enumerate(d):
                        v = (0 if x else 1).to_bytes(dd_n, "little")
                        row = cs[e * cap + slot]
                        if bytes(row[dd_o:dd_o + dd_n]) != v:
                            row[dd_o:dd_o + dd_n] = np.frombuffer(v, np.uint8)
                            cs_dirty = True
                    want = (1 if len(d) and not any(d) else 0).to_bytes(c2_n, "little")
                    if bytes(bs[e, c2_o:c2_o + c2_n]) != want:
                        bs[e, c2_o:c2_o + c2_n] = np.frombuffer(want, np.uint8)
                        bs_dirty = True
                if not np.isnan(frac[i]):
                    cur_frac[e] = float(frac[i])
                dts[e] = float(start_dt[i]) if k else -1.0
            b.set_raw("controls", cb.tobytes())
            if cs_dirty:
                b.set_raw("car_states", cs.tobytes())
            if bs_dirty:
                b.set_raw("ball_states", bs.tobytes())
            b.set_frame_frac(cur_frac)
            b.set_frame_start_dt(dts)
            b.step(1)
            cs = np.frombuffer(bytearray(b.get_raw("car_states")), np.uint8).reshape(N * cap, F.car).copy()
            bs = np.frombuffer(bytearray(b.get_raw("ball_states")), np.uint8).reshape(N, F.ball).copy()
            credited = False
            for e in live:
                seg = segs[batch[e]]
                i = at[seg["start"] + k]
                if i in credits:
                    rows = cs[e * cap:(e + 1) * cap]
                    alive = [j for j in range(cap) if not rows[j, dm_o]]
                    for slot, gain in credits[i]:
                        if len(alive) < cap and slot < len(alive):
                            slot = alive[slot]
                        bst = float(rows[slot, bo_o:bo_o + 4].view("<f4")[0])
                        m = round(gain / 0.12)
                        nb = min(100.0, bst + 12.0 * m) if 1 <= m <= 4 and abs(gain - 0.12 * m) < 1e-3 else 100.0
                        rows[slot, bo_o:bo_o + 4] = np.frombuffer(np.float32(nb).tobytes(), np.uint8)
                    credited = True
            if credited:
                b.set_raw("car_states", cs.tobytes())
            E = [e for e in live if not held[at[segs[batch[e]]["start"] + k]]]
            if E:
                tally_envs([batch[e] for e in E], [accs[e] for e in E], cs.reshape(N, cap, F.car)[E], bs[E],
                           [at[segs[batch[e]]["start"] + k] for e in E], k)
        for e, n in enumerate(batch):
            out[n] = accs[e]

    def tally_envs(ns, accs, cs3, bs, I, k):
        """tally() for several envs at once, with score.py's arithmetic (its Python floats against the truth's
        float32 scalars compute in W) and summation order."""
        def field(car, ball):
            a = np.ascontiguousarray(cs3[:, :ncars, F.c[car][0]:F.c[car][0] + 12]).view("<f4")
            b_ = np.ascontiguousarray(bs[:, F.b[ball][0]:F.b[ball][0] + 12]).view("<f4")[:, None, :]
            return np.concatenate([a, b_], axis=1)
        pos, vel_bt, vel_uu, ang = (field(c, b_) for c, b_ in (("pos_bt", "pos_bt"), ("vel_bt", "vel_bt"),
                                                              ("internal_vel", "vel"), ("internal_ang_vel", "ang_vel")))
        fwd, right, up = (field(c, c) for c in ("basis_forward", "basis_right", "basis_up"))
        T = truth[I][:, :ncars + 1, :18]
        x = np.concatenate([pos, vel_bt, ang, np.stack([fwd, right, up], axis=-1).reshape(len(I), ncars + 1, 9)], -1)
        ok = (x.view(np.uint32) == T.view(np.uint32)).all(-1)

        def dist(a, b):
            d = a.astype(W) - b.astype(W)
            # x ** 2 on numpy scalars is libm pow, not always x * x and not the same on every platform
            d = np.array([v ** 2 for v in d.ravel()], W).reshape(d.shape)
            return np.sqrt((d[..., 0] + d[..., 1] + d[..., 2]).astype(np.float64))
        R = T[..., 9:18].reshape(len(I), ncars + 1, 3, 3)
        terms = [dist(pos, T[..., 0:3]) * S.UU / S.TOL["pos"],
                 dist(vel_uu.astype(np.float64) / S.UU, T[..., 3:6]) * S.UU / S.TOL["vel"],
                 dist(ang, T[..., 6:9]) / S.TOL["ang"], dist(fwd, R[..., :, 0]) / S.TOL["axis"],
                 dist(up, R[..., :, 2]) / S.TOL["axis"]]
        err = np.sqrt(py_sum([t[:, :ncars] * t[:, :ncars] for t in terms] +
                             [np.broadcast_to(t[:, ncars:] * t[:, ncars:], (len(I), ncars)) for t in terms]))
        key = str(k + 1)
        for r, (n, acc) in enumerate(zip(ns, accs)):
            acc[1] += ncars
            acc[0] += int((err[r] < 1.0).sum())
            if key in acc[2]:
                acc[2][key] += [float(v) for v in np.minimum(err[r], 10.0)]
            acc[5] = True
            acc[3] += ncars + 1
            acc[4] += int(ok[r].sum())
            if acc[6] and not ok[r].all():
                acc[6] = False
                j = int(np.argmin(ok[r]))
                acc[7] = [segs[n]["start"], k, "ball" if j == ncars else f"car{j}"]

    batch = []
    for n, seg in enumerate(segs):
        if seg["spawn"]:
            single(n)
            continue
        batch.append(n)
        if len(batch) == BATCH_MAX:
            run(batch)
            batch = []
    if batch:
        run(batch)

    r = dict(name=meta["name"], passed=0, total=0, at_k={str(k): [] for k in S.KS}, exact_bt=0, body_ticks=0,
             exact_segs=0, segs=0, first=[])
    for acc in out:
        r["passed"] += acc[0]
        r["total"] += acc[1]
        for k in r["at_k"]:
            r["at_k"][k] += acc[2][k]
        r["body_ticks"] += acc[3]
        r["exact_bt"] += acc[4]
        if acc[7] is not None and len(r["first"]) < 5:
            r["first"].append(acc[7])
        if acc[5]:
            r["segs"] += 1
            r["exact_segs"] += acc[6]
    return r


if __name__ == "__main__":
    S.score = score
    sys.argv[0] = os.path.join(S.HERE, "score.py")
    S.main()
