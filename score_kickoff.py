"""Score RocketGPU's kickoffs on the full-state suite: where the cars spawn and how long they are held.

    python score_kickoff.py DATA_DIR [--splits dev,test,...] [--names a,b] [--jobs N] [--json OUT]

kickoffs.json lists every kickoff of every recording: the frame the game re-creates the cars ("frame"), the frame it
lets them drive ("release"), and each car's [team, spot] in the game's car order (spot 0-4: the blue-side kickoff spot
index, orange mirrored; -1: a team's sixth car on). A match keeps one order of the five spots; a team's nth car takes
order[(i + n) % 5], where i goes up by one per kickoff. Each recording's order is read off its own kickoffs.

  spots      Arena.reset_to_kickoff() on a fresh arena that holds the match's order and i: each car's spot against
             the game's, with the game's countdown and without it.
  countdown  For a kickoff whose rows are in DATA_DIR/NAME.npz: reset_to_kickoff(game_countdown=True) at the game's
             frame, then the recorded controls and render frames. Every car and the ball against the game on every
             tick up to 30 ticks after the release.
  release    The hold's length: where a car has its throttle down across the release, the bodies on the last held
             tick and the first two free ones are the game's only if the cars were let go on the game's tick.
             The countdown runs on the render frames' time. Recordings made before the recorder kept each frame's
             time ("frame times not recorded") are tallied apart: a frame that dropped time there cannot be replayed.
  settled    reset_to_kickoff() without the countdown: the cars and the ball against the game's as its countdown
             ends, then 30 ticks on the recorded controls. A car whose player held the throttle through the
             countdown has moved in the game and is not compared.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter

import numpy as np

import score as sc

KICKOFF_XY = ((-2048.0, -2560.0), (2048.0, -2560.0), (-256.0, -3840.0), (256.0, -3840.0), (0.0, -4608.0))
AFTER = 30


def spot_of(team, x, y):
    if team:
        x, y = -x, -y
    return next((k for k, (sx, sy) in enumerate(KICKOFF_XY) if abs(sx - x) <= 2.0 and abs(sy - y) <= 2.0), -1)


def match_orders(kicks):
    """Per kickoff (order, i) or None: the match's spot order as far as its kickoffs show it, i counted from the
    recording's first kickoff. A team's cars are taken in the game's car order; with uneven teams only the larger
    team's cars give the order."""
    order, clash = {}, False
    for k, kick in enumerate(kicks):
        teams = [[c for c in kick["cars"] if c[0] == t] for t in (0, 1)]
        most = max(len(t) for t in teams)
        for t in teams:
            if len(t) == most:
                for n, c in enumerate(t[:5]):
                    if c[1] < 0:
                        continue
                    if order.setdefault((k + n) % 5, c[1]) != c[1]:
                        clash = True
    if clash or len(set(order.values())) != len(order):
        return [None] * len(kicks)
    rest = [s for s in range(5) if s not in order.values()]
    full = [order[m] if m in order else rest.pop() for m in range(5)]
    return [(full, k % 5) for k in range(len(kicks))]


def set_match(arena, order, i):
    st = arena.get_full_state(["env_states"])
    packed = 0x8000
    for j, s in enumerate(order):
        packed |= s << (3 * j)
    arena.set_full_state({"env_states.kickoff_match_order": int(packed).to_bytes(2, "little"),
                          "env_states.kickoff_match_index": bytes([i])} if "env_states.kickoff_match_order" in st else {})


def spots_now(cars, teams):
    out = []
    for c, t in zip(cars, teams):
        p = c.get_state().pos
        out.append(spot_of(t, p.x, p.y))
    return out


def rule_check(rp, rs, kick, match, t):
    """reset_to_kickoff on a fresh arena with the kickoff's teams, with and without the countdown."""
    teams = [c[0] for c in kick["cars"]]
    want = [c[1] for c in kick["cars"]]
    for mode, countdown in (("with the countdown", True), ("without the countdown", False)):
        cfg = rp.compat.ArenaConfig()
        arena = rs.Arena(rs.GameMode.SOCCAR, cfg, 120.0)
        cars = [arena.add_car(rs.Team.Orange if tm else rs.Team.Blue, rs.CAR_CONFIG_OCTANE) for tm in teams]
        set_match(arena, *match)
        arena.reset_to_kickoff(countdown)
        if countdown:
            arena.step(2)
        got = spots_now(cars, teams)
        for g, w in zip(got, want):
            if w >= 0:
                t[f"spots {mode}/cars"] += 1
                t[f"spots {mode}/same"] += int(g == w)
            else:
                t[f"spots {mode}/cars past a team's fifth, not compared"] += 1
        t[f"spots {mode}/kickoffs"] += 1
        t[f"spots {mode}/kickoffs same"] += int(all(g == w for g, w in zip(got, want) if w >= 0))


def driving(arena, n):
    st = sc.State(arena)
    fmt = sc.INT_FMT[st.span("car_states.driving_disabled", 0)[1]]
    return [not st.get("car_states.driving_disabled", j, fmt)[0] for j in range(n)]


def replay(rs, arena, cars, z, meta, rows, cols, co, first_scored, t, key):
    """Step the rows with their controls and frames; tally every body against the game from row first_scored on.
    Returns per row whether every body was the game's."""
    ncars = meta["ncars"]
    exact_all, per_row = True, []
    for k, i in enumerate(rows):
        for j, c in enumerate(cars):
            c.set_controls(sc.controls(rs, z["ctl"][i, cols[j]]))
        if not co:
            sc.set_frame(arena, None if np.isnan(z["frac"][i]) else float(z["frac"][i]),
                         float(z["start_dt"][i]) if k else -1.0)
        arena.step(1)
        if k < first_scored:
            per_row.append(True)
            continue
        sim = sc.read_bodies(arena, ncars)
        tr = z["truth"][i][list(cols) + [ncars]]
        row = True
        for j in range(ncars + 1):
            ok = bool(sc.exact(sim[j], tr[j]))
            t[f"{key}/body-ticks"] += 1
            t[f"{key}/exact"] += int(ok)
            row = row and ok
        per_row.append(row)
        exact_all = exact_all and row
    t[f"{key}/kickoffs"] += 1
    t[f"{key}/kickoffs exact"] += int(exact_all)
    return per_row


def run(args):
    path, kicks = args
    rp, rs = sc.build()
    t = Counter()
    name = os.path.basename(path)[:-4]
    matches = match_orders(kicks)
    t["kickoffs"] += len(kicks)
    t["recordings whose kickoffs fit no single order"] += int(bool(kicks) and matches[0] is None)
    for kick, match in zip(kicks, matches):
        if match is None:
            continue
        rule_check(rp, rs, kick, match, t)
    if not os.path.exists(path):
        return dict(name=name, t=dict(t))
    z = np.load(path)
    z = {k: z[k] for k in z.files}
    meta = json.loads(z["meta"].tobytes())
    ncars, co = meta["ncars"], bool(meta.get("controls_only"))
    at = {int(f): i for i, f in enumerate(z["frame"])}
    state = {k[2:]: z[k] for k in z if k.startswith("s.")}
    # the countdown runs on the render frames' time; a recording without the frames' times cannot show a frame that
    # dropped time, and its kickoffs are tallied apart
    timed = co or bool((z["start_dt"] >= 0).any())
    seg_at = {seg["start"]: n for n, seg in enumerate(meta["segments"])
              if any(c[0] == "set_kickoff_spawn_override" for c in seg["calls"])}
    for kick, match in zip(kicks, matches):
        n = seg_at.get(kick["frame"])
        if n is None or match is None or kick["release"] is None or len(kick["cars"]) != ncars:
            t["kickoffs without rows"] += 1
            continue
        seg = meta["segments"][n]
        frames = range(kick["frame"], kick["release"] + AFTER)
        if any(f not in at for f in frames):
            t["kickoffs without rows"] += 1
            continue
        rows = [at[f] for f in frames]
        cols = list(seg["perm"]) if co else list(range(ncars))
        hold = next(k for k, i in enumerate(rows) if not z["held"][i])
        start = {k: v[n].tobytes() for k, v in state.items()}

        arena, cars = sc.restore(rp, rs, seg, meta["cars"][seg["cars"]], start)
        set_match(arena, *match)
        arena.reset_to_kickoff(True)
        key = "" if timed else ", frame times not recorded"
        per_row = replay(rs, arena, cars, z, meta, rows, cols, co, 1, t, "countdown" + key)
        # the release tick shows in the bodies when a car has its throttle down on the last held tick and the first
        # free one: let go a tick early or late, it would not be where the game's car is
        if any(z["ctl"][rows[hold - 1], c, 0] and z["ctl"][rows[hold], c, 0] for c in cols):
            t[f"release{key}/kickoffs with a car on the throttle across the release"] += 1
            t[f"release{key}/on the game's tick"] += int(per_row[hold - 1] and per_row[hold] and per_row[hold + 1])

        arena, cars = sc.restore(rp, rs, seg, meta["cars"][seg["cars"]], start)
        set_match(arena, *match)
        arena.reset_to_kickoff(False)
        t["settled/kickoffs"] += 1
        t["settled/cars free at once"] += int(all(driving(arena, ncars)))
        sim = sc.read_bodies(arena, ncars)
        tr = z["truth"][rows[hold - 1]][cols + [ncars]]
        idle = [not np.any(z["ctl"][rows[:hold], cols[j], 0]) for j in range(ncars)]
        for j in range(ncars + 1):
            if j < ncars and not idle[j]:
                t["settled/cars not compared (throttle held in the countdown)"] += 1
                continue
            t["settled/bodies"] += 1
            t["settled/exact"] += int(bool(sc.exact(sim[j], tr[j])))
        if all(idle):
            replay(rs, arena, cars, z, meta, rows[hold:], cols, co, 0, t, "settled, 30 ticks on")
    return dict(name=name, t=dict(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--splits", default="all")
    ap.add_argument("--names")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--json")
    a = ap.parse_args()
    if not hasattr(sc.build()[0].compat.Arena, "reset_to_kickoff"):
        raise SystemExit("this RocketGPU build has no Arena.reset_to_kickoff(): score_kickoff.py needs a build later than 1.3.0")
    here = os.path.dirname(os.path.abspath(__file__))
    man = json.load(open(os.path.join(here, "manifest.json")))["recordings"]
    kicks = json.load(open(os.path.join(here, "kickoffs.json")))
    want = None if a.splits == "all" else set(a.splits.split(","))
    names = a.names.split(",") if a.names else [r["name"] for r in man if want is None or r["split"] in want]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        res = pool.map(run, [(os.path.join(a.data, n + ".npz"), kicks.get(n, [])) for n in names], chunksize=1)
    tot = Counter()
    for r in res:
        tot.update(r["t"])
    print(f"{len(res)} recordings, {tot['kickoffs']} kickoffs, {time.time() - t0:.0f} s")
    for k in sorted(tot):
        if k != "kickoffs":
            print(f"  {k}: {tot[k]}")
    if a.json:
        json.dump(dict(summary=dict(tot), recordings=res), open(a.json, "w"))


if __name__ == "__main__":
    main()
