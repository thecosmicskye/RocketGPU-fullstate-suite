"""Score RocketGPU's state APIs on the full-state suite: what a car's state reports after each tick, against the game.

    python score_api.py DATA_DIR [--splits dev,test,...] [--names a,b] [--jobs N] [--json OUT]

Replays the segments as score.py does and compares, per car and tick, with DATA_DIR/NAME.api.npz:

  native  rocketgpu::CarState (Car.get_native_state()) against the game's own memory. Flags the tick's physics sets
          are read right after the tick; is_on_ground, has_jumped and the double-jump flag are the game's as the next
          tick finds them, before its own input (the game refreshes them when a render frame starts, so ticks in the
          middle of a render frame are not scored).
  rlbot   rocketgpu/rlbotv5.h (the rlbotv5_* methods) against the RLBot v5 packet logged for the tick, bit for bit.
          RLBot keeps its jump flags and dodge timers itself, and a restored state does not hold them: those fields
          are scored once the car has been at rest on the ground in the segment (no jump or dodge flag or timer set
          in the packet or the car), and again after any tick RLBot sent no packet for (one in the middle of a
          render frame): its own clocks skip there. RLBot's boost count starts from the packet the same way.
          RLBot shows the boost of a pad that came back under a waiting car one tick late. A controls-only recording
          leaves the pads to the build, so the build decides which pickups those are. The other recordings give the
          game's own pad credits, and there the packet says whether a credit showed on its tick or the next.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter

import numpy as np

import score as sc

F32 = np.float32
WHEELS = (1, 0, 3, 2)   # RocketGPU's wheel k is the game's wheel WHEELS[k]
NATIVE = ("is_on_ground", "wheels_with_contact", "has_jumped", "has_double_jumped | has_flipped", "is_jumping",
          "is_flipping", "is_boosting", "is_supersonic", "is_auto_flipping", "handbrake_val", "boost",
          "world_contact.has_contact")
RLBOT_BODY = ("physics.location", "physics.rotation", "physics.velocity", "physics.angular_velocity", "is_supersonic",
              "boost")
RLBOT_JUMP = ("air_state", "has_jumped", "has_double_jumped", "has_dodged", "dodge_timeout", "dodge_elapsed")
RLBOT_BALL = ("ball physics.location", "ball physics.rotation", "ball physics.velocity",
              "ball physics.angular_velocity")


def native_truth(a, i, c):
    """The game's values for column c after row i (see the module text)."""
    xt = a["game_e_times"][i, c]
    jump_starts = a["game_e_jump_active"][i, c] and xt[0] == 0.0
    air_starts = (a["game_e_double_jump_active"][i, c] and xt[1] == 0.0) or \
                 (a["game_e_dodge_active"][i, c] and xt[2] == 0.0)
    gw = a["game_x_wheels"][i, c]
    return (int(a["game_e_on_ground"][i, c]), tuple(int(gw[w]) for w in WHEELS),
            int(a["game_x_jumped"][i, c] and a["game_e_jumped"][i, c] and not jump_starts),
            int(a["game_x_double_jumped"][i, c] and a["game_e_double_jumped"][i, c] and not air_starts),
            int(a["game_x_jump_active"][i, c]), int(a["game_x_dodge_active"][i, c]), int(a["game_x_boost_active"][i, c]),
            int(a["game_x_supersonic"][i, c]), int(a["game_x_flip_over_active"][i, c]),
            F32(a["game_x_handbrake"][i, c]), F32(F32(a["game_e_boost"][i, c]) * F32(100.0)),
            int(a["game_x_world_contact"][i, c]))


def native_ours(s):
    return (int(s.is_on_ground), tuple(int(w) for w in s.wheels_with_contact), int(s.has_jumped),
            int(bool(s.has_double_jumped or s.has_flipped)), int(s.is_jumping), int(s.is_flipping), int(s.is_boosting),
            int(s.is_supersonic), int(s.is_auto_flipping), F32(s.handbrake_val), F32(s.boost),
            int(s.world_contact.has_contact))


def phys_rows(p):
    return (np.array([p.location.x, p.location.y, p.location.z], F32), np.array([p.pitch, p.yaw, p.roll], F32),
            np.array([p.velocity.x, p.velocity.y, p.velocity.z], F32),
            np.array([p.angular_velocity.x, p.angular_velocity.y, p.angular_velocity.z], F32))


def set_start_boost(arena, amounts):
    """Builds whose car state holds the game's boost amount (boost_amount, 0..1) start from the game's own."""
    st = sc.State(arena)
    if "car_states.boost_amount" not in st.st:
        return
    for j, g in enumerate(amounts):
        if not np.isnan(g):
            st.put("car_states.boost_amount", j, "<f", float(g))
    st.commit()


def credit_boost(arena, credits):
    """score.py's credit_boost in the game's own arithmetic, for builds that hold boost_amount: a small pad adds
    0.12, a big one fills."""
    st = sc.State(arena)
    if "car_states.boost_amount" not in st.st:
        return sc.credit_boost(arena, credits)
    n = st.cars
    live = [j for j in range(n) if not st.get("car_states.is_demoed", j, "<B")[0]]
    for slot, gain in credits:
        if len(live) < n and slot < len(live):
            slot = live[slot]
        g, b = F32(st.get("car_states.boost_amount", slot, "<f")[0]), F32(st.get("car_states.boost", slot, "<f")[0])
        if F32(g * F32(100.0)) != b:    # a boost written without its amount: the build takes boost / 100
            g = F32(b / F32(100.0))
        m = round(gain / 0.12)
        if 1 <= m <= 4 and abs(gain - 0.12 * m) < 1e-3:
            for _ in range(m):
                g = min(F32(1.0), F32(g + F32(0.12)))
        else:
            g = F32(1.0)
        st.put("car_states.boost_amount", slot, "<f", float(g))
        st.put("car_states.boost", slot, "<f", float(F32(g * F32(100.0))))
    st.commit()


def boost_sets(arena, a, i, cols, credited=()):
    """A scripted recording sets boost between ticks. Where the game's amount moves between a tick's end and the next
    tick's start for a car with no pad credit on that tick, the arena's car gets the game's amount; returns those
    cars."""
    st, out = None, []
    for j, c in enumerate(cols):
        if j not in credited and a["game_ok"][i, c] and a["game_e_boost"][i, c] != a["game_x_boost"][i, c]:
            st = st or sc.State(arena)
            g = F32(a["game_e_boost"][i, c])
            if "car_states.boost_amount" in st.st:
                st.put("car_states.boost_amount", j, "<f", float(g))
            st.put("car_states.boost", j, "<f", float(F32(g * F32(100.0))))
            out.append(j)
    if st:
        st.commit()
    return out


class Tally:
    def __init__(self):
        self.n, self.same, self.first = Counter(), Counter(), {}
        self.cars = Counter()   # per layer: car-ticks scored, and those with every scored field equal

    def add(self, key, same, where):
        self.n[key] += 1
        self.same[key] += int(bool(same))
        if not same and len(self.first.setdefault(key, [])) < 3:
            self.first[key].append(where)

    def result(self, name):
        return dict(name=name, n={"/".join(k): v for k, v in self.n.items()},
                    same={"/".join(k): v for k, v in self.same.items()},
                    first={"/".join(k): v for k, v in self.first.items()}, cars=dict(self.cars))


def tick(rp, t, a, i, k, cars, cols, arena, mut, grounded, boost, where0, mid_frame, sets=(), credited={}):
    """Tally one tick: arena car j is column cols[j] of the data. sets: cars whose boost the recording's script set
    after this tick (boost_sets). credited: car -> its boost before the pad credit the recording gave it this tick."""
    ncars = len(cars)
    states = [c.get_native_state() for c in cars]
    for j, s in enumerate(states):
        c = cols[j]
        where = where0 + [k, j]
        if j in sets:
            t.cars["native/boost set by the recording's script, not scored"] += 1
            boost[j] = None
        if a["game_ok"][i, c] and not s.is_demoed:
            if mid_frame:
                t.cars["native/mid-frame, not scored"] += 1
            else:
                ours, game = native_ours(s), native_truth(a, i, c)
                all_same = True
                for name, o, g in zip(NATIVE, ours, game):
                    same = o == g or (name == "boost" and j in sets)
                    all_same = all_same and same
                    t.add(("native", name), same, where + [repr(o), repr(g)])
                t.cars["native/car-ticks"] += 1
                t.cars["native/exact"] += int(all_same)
        if a["rlbot_car_ok"][i, c] and not s.is_demoed:
            p = a["rlbot_cars"][i, c]
            ours = phys_rows(s.rlbotv5_physics(mut))
            all_same = True
            for name, o, g in zip(RLBOT_BODY, ours, (p[0:3], p[3:6], p[6:9], p[9:12])):
                same = np.array_equal(o, g)
                all_same = all_same and same
                t.add(("rlbot", name), same, where + [o.tolist(), g.tolist()])
            same = int(s.rlbotv5_is_supersonic(mut)) == int(p[20])
            all_same = all_same and same
            t.add(("rlbot", "is_supersonic"), same, where)
            if j in sets or (boost[j] is None and j in credited):
                pass   # this packet can be from before the script's set or the pad's credit: the count starts on the next
            elif boost[j] is None:   # RLBot counts boost itself: its count starts from the packet
                boost[j] = rp.RlbotV5Boost()
                boost[j].boost, boost[j].whole = float(p[12]), int(np.ceil(s.boost))
            else:
                o = None
                if j in credited and hasattr(s, "rlbotv5_boost_hold"):
                    # RLBot shows the boost of a pad that came back under the car a tick late. A controls-only
                    # recording leaves the pads to the build; here the recording gives the credit, so which of the
                    # two it was is read from the packet.
                    now, late = rp.RlbotV5Boost(), rp.RlbotV5Boost()
                    now.boost, now.whole = late.boost, late.whole = boost[j].boost, boost[j].whole
                    if F32(now.update(s)) != F32(p[12]):
                        s.rlbotv5_boost_hold = int(np.ceil(credited[j])) + 1
                        o = F32(late.update(s))
                        s.rlbotv5_boost_hold = 0
                        if o == F32(p[12]):
                            boost[j] = late
                            t.cars["rlbot/pad credits shown a tick later (read from the packet)"] += 1
                        else:
                            o = None
                if o is None:
                    o = F32(boost[j].update(s))
                same = o == F32(p[12])
                all_same = all_same and same
                t.add(("rlbot", "boost"), same, where + [float(o), float(p[12])])
            if not grounded[j] and not np.any(p[13:17]) and p[17] < 0 and p[18] == 0 and s.is_on_ground and \
                    not s.is_jumping and not s.has_jumped and not s.has_double_jumped and not s.has_flipped:
                grounded[j] = True
                if s.rlbotv5_has_jumped():   # RLBot's own has_jumped is not in the recorded state: it starts here
                    st = sc.State(arena)
                    st.put("car_states.game_has_jumped", j, "<B", 0)
                    st.commit()
                    s = cars[j].get_native_state()
            if grounded[j]:
                ours = (int(s.air_state), int(s.rlbotv5_has_jumped()), int(s.rlbotv5_has_double_jumped()),
                        int(s.rlbotv5_has_dodged()), F32(s.rlbotv5_dodge_timeout()), F32(s.rlbotv5_dodge_elapsed()))
                game = (int(p[13]), int(p[14]), int(p[15]), int(p[16]), F32(p[17]), F32(p[18]))
                for name, o, g in zip(RLBOT_JUMP, ours, game):
                    same = o == g
                    all_same = all_same and same
                    t.add(("rlbot", name), same, where + [repr(o), repr(g)])
            else:
                t.cars["rlbot/jump fields not scored (car not yet at rest on the ground in the segment)"] += 1
            t.cars["rlbot/car-ticks"] += 1
            t.cars["rlbot/exact"] += int(all_same)
        else:
            # RLBot sent nothing for the car on this tick (a tick in the middle of a render frame, a demolished car):
            # the values it counts itself restart
            boost[j] = None
            grounded[j] = False
    if a["rlbot_ok"][i] and not np.isnan(a["rlbot_ball"][i, 0]):
        b = a["rlbot_ball"][i]
        ours = phys_rows(arena.ball.get_native_state().rlbotv5_physics(mut))
        for name, o, g in zip(RLBOT_BALL, ours, (b[0:3], b[3:6], b[6:9], b[9:12])):
            t.add(("rlbot", name), np.array_equal(o, g), where0 + [k, o.tolist(), g.tolist()])


def run(path):
    rp, rs = sc.build()
    z = np.load(path)
    z = {k: z[k] for k in z.files}
    a = np.load(path[:-4] + ".api.npz")
    a = {k: a[k] for k in a.files}
    meta = json.loads(z["meta"].tobytes())
    ncars = meta["ncars"]
    at = {int(f): i for i, f in enumerate(z["frame"])}
    t = Tally()
    state = {k[2:]: z[k] for k in z if k.startswith("s.")}
    co = bool(meta.get("controls_only"))
    credits = {}
    if not co:
        for j, s, g in zip(z["credit_tick"], z["credit_slot"], z["credit_gain"]):
            credits.setdefault(int(j), []).append((int(s), float(g)))
    for n, seg in enumerate(meta["segments"]):
        arena, cars = sc.restore(rp, rs, seg, meta["cars"][seg["cars"]], {k: v[n].tobytes() for k, v in state.items()})
        if "game_start_boost" in a:
            set_start_boost(arena, a["game_start_boost"][n])
        mut = arena.get_mutator_config()
        cols = list(seg["perm"]) if co else list(range(ncars))
        grounded, boost = [False] * ncars, [None] * ncars
        for k in range(seg["ticks"]):
            i = at[seg["start"] + k]
            for j, c in enumerate(cars):
                c.set_controls(sc.controls(rs, z["ctl"][i, cols[j]]))
            if not co:
                if not seg["timer_hold"]:
                    sc.set_driving(arena, [int(d) for d in z["drv"][i]])
                sc.set_frame(arena, None if np.isnan(z["frac"][i]) else float(z["frac"][i]),
                             float(z["start_dt"][i]) if k else -1.0)
            arena.step(1)
            sets, credited = (), {}
            if i in credits:
                before = [float(c.get_native_state().boost) for c in cars]
                credit_boost(arena, credits[i])
                credited = {j: b for j, (b, c) in enumerate(zip(before, cars)) if c.get_native_state().boost != b}
            if not co and not z["frac"][i] < 0:
                sets = boost_sets(arena, a, i, cols, credited)
            if seg["spawn"] and k == 0 or z["held"][i]:
                boost = [None] * ncars
                continue
            tick(rp, t, a, i, k, cars, cols, arena, mut, grounded, boost, [meta["name"], seg["start"]],
                 not co and z["frac"][i] < 0, sets, credited)
    return t.result(meta["name"])


def report(res):
    tot = Tally()
    for r in res:
        for d, into in ((r["n"], tot.n), (r["same"], tot.same), (r["cars"], tot.cars)):
            for k, v in d.items():
                into[k] += v
        for k, v in r["first"].items():
            tot.first.setdefault(k, [])
            tot.first[k] += v[:3 - len(tot.first[k])]
    for layer, fields in (("native", NATIVE), ("rlbot", RLBOT_BODY + RLBOT_JUMP + RLBOT_BALL)):
        n, ex = tot.cars[f"{layer}/car-ticks"], tot.cars[f"{layer}/exact"]
        print(f"[{layer}] {n} car-ticks, every field equal on {ex} ({100.0 * ex / max(n, 1):.4f}%)")
        for f in fields:
            k = f"{layer}/{f}"
            if tot.n[k]:
                miss = tot.n[k] - tot.same[k]
                print(f"  {f:32} {tot.same[k]:>9} of {tot.n[k]:>9}" +
                      (f"   {miss} differ, first (recording, start, tick, car, ours, game): {tot.first.get(k, [])[:2]}"
                       if miss else ""))
        for k in sorted(tot.cars):
            if k.startswith(layer + "/") and k.split("/", 1)[1] not in ("car-ticks", "exact"):
                print(f"  {k.split('/', 1)[1]}: {tot.cars[k]}")
    return dict(n=dict(tot.n), same=dict(tot.same), cars=dict(tot.cars), first=tot.first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--splits", default="all")
    ap.add_argument("--names")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--json")
    a = ap.parse_args()
    if not hasattr(sc.build()[0].compat.Car, "get_native_state"):
        raise SystemExit("this RocketGPU build has no Car.get_native_state(): score_api.py needs a build later than 1.3.0")
    man = json.load(open(os.path.join(sc.HERE, "manifest.json")))["recordings"]
    splits = list(dict.fromkeys(x["split"] for x in man)) if a.splits == "all" else a.splits.split(",")
    names = a.names.split(",") if a.names else [x["name"] for x in man if x["split"] in splits]
    names = [n for n in names if os.path.exists(os.path.join(a.data, n + ".api.npz"))]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        res = pool.map(run, [os.path.join(a.data, n + ".npz") for n in names], chunksize=1)
    print(f"{len(names)} recordings, {time.time() - t0:.0f} s")
    out = report(res)
    if a.json:
        json.dump(dict(summary=out, recordings=res), open(a.json, "w"))


if __name__ == "__main__":
    main()
