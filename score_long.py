"""Long runs: score.py's segments chained, with no state restored on the way.

    python score_long.py DATA_DIR [--every N] [--max-ticks T] [--splits S] [--jobs J] [--json OUT]

From every N-th segment's start state the arena keeps stepping on the recording's inputs across the following
segments, and every body is compared with the game's after every tick. A run ends, still exact, where the next tick
is not recorded, is held (not scored), or belongs to a segment that starts with a spawn, a timer hold or another car
setup: the game re-creates cars there. --max-ticks ends it earlier (0: no limit). A recording's runs are stepped
together as the envs of one DevBatch, as score_batch.py steps its segments; runs that start with a spawn run on a
single arena. The controls-only recordings are left out.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import score as S
import score_batch as SB


def run_file(job):
    path, every, max_ticks = job
    rp, rs = S.build()
    F = SB.Fields(rp)
    z = np.load(path)
    meta = json.loads(z["meta"].tobytes())
    if meta.get("controls_only"):
        return []
    ncars = meta["ncars"]
    state = {k[2:]: z[k] for k in z.files if k.startswith("s.")}
    ctl, drv, frac, start_dt, held, truth = z["ctl"], z["drv"], z["frac"], z["start_dt"], z["held"], z["truth"]
    at = {int(f): i for i, f in enumerate(z["frame"])}
    credits = {}
    for j, s, g in zip(z["credit_tick"], z["credit_slot"], z["credit_gain"]):
        credits.setdefault(int(j), []).append((int(s), float(g)))
    segs = meta["segments"]
    owner = {}  # tick -> segment index
    for n, seg in enumerate(segs):
        for k in range(seg["ticks"]):
            owner.setdefault(seg["start"] + k, n)

    def plan(n):
        """The run's ticks from the recording alone: each tick's row, whether its driving flags are written, and
        why the run ends."""
        seg, rows, drive, stop = segs[n], [], [], "limit"
        for k in range(max_ticks or len(at)):
            t = seg["start"] + k
            o = owner.get(t)
            if t not in at or o is None:
                stop = "gap"
                break
            cur = segs[o]
            if o != n and cur["start"] == t and (cur["spawn"] or cur["timer_hold"] or cur["cars"] != seg["cars"]):
                stop = "event"
                break
            i = at[t]
            if k and held[i]:
                stop = "held"
                break
            rows.append(i)
            drive.append(not cur["timer_hold"])
        else:
            if not max_ticks:
                stop = "gap"
        return rows, drive, stop

    def result(n, ran, stop, first):
        compared = ran - (1 if segs[n]["spawn"] and ran else 0)
        return dict(rec=meta["name"], seg=n, start=segs[n]["start"], ran=ran, body_ticks=compared * (ncars + 1),
                    stop=stop if first is None else "inexact", first=first)

    def single(n, rows, drive, stop):
        seg = segs[n]
        arena, cars = S.restore(rp, rs, seg, meta["cars"][seg["cars"]], {k: v[n].tobytes() for k, v in state.items()})
        first, ran = None, 0
        for k, i in enumerate(rows):
            for slot, c in enumerate(cars):
                c.set_controls(S.controls(rs, ctl[i, slot]))
            if drive[k]:
                S.set_driving(arena, [int(d) for d in drv[i]])
            S.set_frame(arena, None if np.isnan(frac[i]) else float(frac[i]), float(start_dt[i]) if k else -1.0)
            arena.step(1)
            if i in credits:
                S.credit_boost(arena, credits[i])
            ran = k + 1
            if seg["spawn"] and k == 0:
                continue
            sim = S.read_bodies(arena, ncars)
            bad = [j for j in range(ncars + 1) if not S.exact(sim[j], truth[i][j])]
            if bad:
                first = (k, ["ball" if j == ncars else f"car{j}" for j in bad])
                break
        return result(n, ran, stop, first)

    def batched(runs):
        """runs: (n, rows, drive, stop) of non-spawn starts whose arenas share a mutator config. score_batch.py's
        tick loop without its segment ends."""
        b, cur_frac = None, []
        for e, (n, _, _, _) in enumerate(runs):
            seg = segs[n]
            arena, cars = SB.restore_into(rp, rs, seg, meta["cars"][seg["cars"]],
                                          {k: v[n].tobytes() for k, v in state.items()})
            if b is None:
                b = rp.compat.DevBatch(arena, len(runs))
            b.set_env(e, arena)
            cur_frac.append(arena.get_mutator_config().boost_pad_frame_frac)
        b.commit()
        cap, N = b.car_capacity, len(runs)
        slots = [len(meta["cars"][segs[n]["cars"]]) for n, _, _, _ in runs]
        first, ran = [None] * N, [0] * N
        cs = np.frombuffer(bytearray(b.get_raw("car_states")), np.uint8).reshape(N * cap, F.car).copy()
        bs = np.frombuffer(bytearray(b.get_raw("ball_states")), np.uint8).reshape(N, F.ball).copy()
        dd_o, dd_n = F.c["driving_disabled"]
        c2_o, c2_n = F.b["constrained_2d"]
        bo_o, dm_o = F.c["boost"][0], F.c["is_demoed"][0]
        cb = np.zeros(N * cap, SB.CTL)

        def field(cs3, bs_, car, ball):
            a = np.ascontiguousarray(cs3[:, :ncars, F.c[car][0]:F.c[car][0] + 12]).view("<f4")
            b_ = np.ascontiguousarray(bs_[:, F.b[ball][0]:F.b[ball][0] + 12]).view("<f4")[:, None, :]
            return np.concatenate([a, b_], axis=1)

        for k in range(max(len(r[1]) for r in runs)):
            live = [e for e, r in enumerate(runs) if k < len(r[1]) and first[e] is None]
            if not live:
                break
            cs_dirty = bs_dirty = False
            dts = [-1.0] * N
            for e in live:
                i = runs[e][1][k]
                for slot in range(slots[e]):
                    c = ctl[i, slot]
                    cb[e * cap + slot] = ((float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4])),
                                          (bool(c[5]), bool(c[6]), bool(c[7])), 0)
                if runs[e][2][k]:
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
                i = runs[e][1][k]
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
            cs3, bsl = cs.reshape(N, cap, F.car)[live], bs[live]
            pos, vel, ang = (field(cs3, bsl, c, b_) for c, b_ in (("pos_bt", "pos_bt"), ("vel_bt", "vel_bt"),
                                                                 ("internal_ang_vel", "ang_vel")))
            fwd, right, up = (field(cs3, bsl, c, c) for c in ("basis_forward", "basis_right", "basis_up"))
            x = np.concatenate([pos, vel, ang, np.stack([fwd, right, up], axis=-1).reshape(len(live), ncars + 1, 9)],
                               -1)
            T = np.ascontiguousarray(truth[[runs[e][1][k] for e in live]][:, :ncars + 1, :18], np.float32)
            ok = (x.view(np.uint32) == T.view(np.uint32)).all(-1)
            for r, e in enumerate(live):
                ran[e] = k + 1
                if not ok[r].all():
                    first[e] = (k, ["ball" if j == ncars else f"car{j}" for j in range(ncars + 1) if not ok[r, j]])
        return [result(n, ran[e], stop, first[e]) for e, (n, _, _, stop) in enumerate(runs)]

    out, groups = {}, {}
    for n in range(0, len(segs), every):
        rows, drive, stop = plan(n)
        if segs[n]["spawn"] or not rows:
            out[n] = single(n, rows, drive, stop)
        else:
            m = {k: v for k, v in segs[n]["mutator"].items() if k not in ("boost_pad_frame_frac", "frame_start_dt")}
            groups.setdefault(json.dumps(m, sort_keys=True), []).append((n, rows, drive, stop))
    for runs in groups.values():
        for i in range(0, len(runs), SB.BATCH_MAX):
            for r in batched(runs[i:i + SB.BATCH_MAX]):
                out[r["seg"]] = r
    return [out[n] for n in sorted(out)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--every", type=int, default=10, help="a run from every N-th segment")
    ap.add_argument("--max-ticks", type=int, default=0, help="longest run in ticks; 0: to the run's own end")
    ap.add_argument("--splits", default="all", help="comma-separated splits from manifest.json, or all")
    ap.add_argument("--jobs", type=int, default=4, help="worker processes")
    ap.add_argument("--json")
    a = ap.parse_args()
    man = json.load(open(os.path.join(S.HERE, "manifest.json")))["recordings"]
    splits = [r["split"] for r in man] if a.splits == "all" else a.splits.split(",")
    names = [r["name"] for r in man if r["split"] in splits]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        res = [r for rs_ in pool.imap_unordered(run_file, [(os.path.join(a.data, n + ".npz"), a.every, a.max_ticks)
                                                           for n in names], chunksize=1) for r in rs_]
    print(f"{len(names)} recordings, {time.time() - t0:.0f} s")
    bad = [r for r in res if r["stop"] == "inexact"]
    ran = [r["ran"] for r in res]
    print(f"runs {len(res)}, exact to their end {len(res) - len(bad)}, inexact {len(bad)}")
    print(f"  ticks per run: mean {np.mean(ran):.0f}, longest {max(ran)};  body-ticks compared "
          f"{sum(r['body_ticks'] for r in res)}")
    ends = {s: sum(r["stop"] == s for r in res) for s in ("gap", "event", "held", "limit")}
    print(f"  ended by: the recording's end or a gap {ends['gap']}, a spawn, timer hold or car change {ends['event']}, "
          f"a held tick {ends['held']}, --max-ticks {ends['limit']}")
    for r in sorted(bad, key=lambda r: (r["rec"], r["seg"])):
        print(f"  INEXACT {r['rec']} seg {r['seg']} start {r['start']}: tick {r['first'][0]} {r['first'][1]}")
    if a.json:
        json.dump(res, open(a.json, "w"))


if __name__ == "__main__":
    main()
