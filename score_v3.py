"""Score RocketSim v3 on the full-state suite: the same segments, inputs, pad credits and metrics as score.py.

    python score_v3.py DATA_DIR WORKER [--splits dev,test,...] [--jobs N] [--json OUT]

WORKER is the v3/ build (target/release/fs_v3); it needs ROCKETGPU_COLLISION_MESH_DIR. v3 takes no hidden state, so
each segment starts from the public start state (the data's "public" rows: per car the preset, team and v3 CarState
fields, then the ball, whether it sleeps, and the boost pads).
"""
import argparse
import json
import multiprocessing as mp
import os
import struct
import subprocess
import time

import numpy as np

import score as sc

CAR_F = 2 + 59        # preset, team, then fs_v3's CAR_F state floats
DEMOED = 2 + 52       # is_demoed in a car's row
F32 = np.float32


def run(args):
    path, worker = args
    z = np.load(path)
    meta = json.loads(z["meta"].tobytes())
    ncars = meta["ncars"]
    ctl, drv, held, truth, pub = z["ctl"], z["drv"], z["held"], z["truth"], z["public"]
    at = {int(f): i for i, f in enumerate(z["frame"])}
    credits = {}
    for j, s, g in zip(z["credit_tick"], z["credit_slot"], z["credit_gain"]):
        credits.setdefault(int(j), []).append((int(s), float(g)))
    w = subprocess.Popen([worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    r = dict(name=meta["name"], passed=0, total=0, at_k={str(k): [] for k in sc.KS}, exact_bt=0, body_ticks=0,
             exact_segs=0, segs=0, first=[])
    for n, seg in enumerate(meta["segments"]):
        start = list(pub[n])
        live = [j for j in range(ncars) if not start[j * CAR_F + DEMOED]]
        ticks, rows = [], []
        for k in range(seg["ticks"]):
            i = at[seg["start"] + k]
            gain = [0.0] * ncars
            for slot, g in credits.get(i, ()):
                if len(live) < ncars and slot < len(live):
                    slot = live[slot]
                gain[slot] = g
            for j in range(ncars):
                ticks += list(ctl[i, j]) + [bool(drv[i, j]), gain[j]]
            rows.append(None if seg["spawn"] and k == 0 or held[i] else i)
        msg = [ncars, seg["ticks"]] + start + ticks
        w.stdin.write(struct.pack("<%dd" % len(msg), *map(float, msg)))
        w.stdin.flush()
        nb = ncars + 1
        out = np.frombuffer(w.stdout.read(4 * 18 * nb * len(rows)), F32).reshape(len(rows), nb, 18)
        seg_exact, scored = True, False
        for k, i in enumerate(rows):
            if i is None:
                continue
            sim = [dict(pos=o[0:3] / F32(50), vel=o[3:6] / F32(50), vel_bt=o[3:6] / F32(50), ang=o[6:9],
                        fwd=o[9:12], right=o[12:15], up=o[15:18]) for o in out[k]]
            for e in sc.norms(sim, sc.truth_rows(truth[i]), ncars):
                r["total"] += 1
                r["passed"] += e < 1.0
                if str(k + 1) in r["at_k"]:
                    r["at_k"][str(k + 1)].append(min(e, 10.0))
            scored = True
            for j in range(nb):
                ok = sc.exact(sim[j], truth[i][j])
                r["body_ticks"] += 1
                r["exact_bt"] += ok
                seg_exact = seg_exact and ok
        if scored:
            r["segs"] += 1
            r["exact_segs"] += seg_exact
    w.stdin.write(struct.pack("<d", -1.0))
    w.stdin.close()
    w.wait()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("worker")
    ap.add_argument("--splits", default="all")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--json")
    a = ap.parse_args()
    man = json.load(open(os.path.join(sc.HERE, "manifest.json")))["recordings"]
    split_of = {x["name"]: x["split"] for x in man}
    splits = list(dict.fromkeys(x["split"] for x in man)) if a.splits == "all" else a.splits.split(",")
    names = [x["name"] for x in man if x["split"] in splits]
    worker = os.path.abspath(a.worker)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        res = pool.map(run, [(os.path.join(a.data, n + ".npz"), worker) for n in names], chunksize=1)
    print(f"{len(names)} recordings, {time.time() - t0:.0f} s")
    groups = {s: {r["name"] for r in res if split_of[r["name"]] == s} for s in splits}
    groups["all"] = set(names)
    out = sc.summary(res, groups)
    sc.report(out, [])
    if a.json:
        json.dump(dict(summary=out, recordings=res), open(a.json, "w"))


if __name__ == "__main__":
    main()
