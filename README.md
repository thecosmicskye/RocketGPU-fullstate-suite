# RocketGPU full-state suite

144 Rocket League recordings with the game's full physics state, and the scripts that score
[RocketGPU](https://github.com/thecosmicskye/RocketGPU) and RocketSim v3 against them.

Each segment starts from the game's state at one tick and runs open loop for up to 120 ticks on the
game's recorded inputs. Each tick, every car and the ball are compared with the game:

- **exact**: position, velocity, angular velocity and rotation all equal to the game's, bit for bit.
- **segment pass**: car-ticks whose normalized error is below 1 (position / 2 uu, velocity / 3 uu/s,
  angular velocity / 0.05 rad/s, forward and up axes / 0.05, over the car and the ball).
- **mean error**: that error, capped at 10, at 1, 30 and 120 ticks.

## Results

13,930,650 body-ticks in 25,622 segments.

| | RocketGPU (CUDA) | RocketGPU (CPU) | RocketSim v3 0.3.5 |
|---|---|---|---|
| Exact body-ticks | 100.0000% | 100.0000% | 4.3193% |
| Exact segments | 100.000% | 100.000% | 0.000% |
| Segment pass | 100.000% | 100.000% | 86.154% |
| Mean error k1 / k30 / k120 | 0 / 0 / 0 | 0 / 0 / 0 | 0.80 / 0.65 / 1.63 |

RocketGPU starts from the full state (bodies, broadphase, contact manifolds, timers). RocketSim v3
starts from the public state (what its `CarState`, `BallState` and boost pad states hold).

## Data

Recorded in Rocket League. The truth is the game's own state at each tick. Each segment's start state is the
game's state at that tick in RocketGPU's state fields (the names `Arena.get_full_state()` uses), so it loads
with `set_full_state` in
[RocketGPU 15484501](https://github.com/thecosmicskye/RocketGPU/commit/15484501768d0a537eed853e3287f0d93ea1b417)
or later builds with the same fields.

The recordings are a release asset (878 MB):

```bash
gh release download v3 -R thecosmicskye/RocketGPU-fullstate-suite
```

```bash
tar xf fullstate-suite-v3.tar && sha256sum -c data.sha256
```

`data/NAME.npz` holds, per recording:

| Array | |
|---|---|
| `frame`, `ctl`, `drv`, `frac`, `start_dt`, `held` | per tick: game tick, each car's controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake), whether the game lets it drive, the render frame's interpolation fraction and start time, all cars held (not scored) |
| `truth` | per tick, each car then the ball: position and velocity (Bullet units, uu / 50), angular velocity, rotation matrix by rows |
| `credit_tick`, `credit_slot`, `credit_gain` | the game's boost pad pickups |
| `s.*`, `meta` | per segment, the full start state (`Arena.get_full_state()` fields), and the cars, boost pads and mutators |
| `public` | per segment, the public start state (the rows `score_v3.py` sends) |

The `uneven` and `pads` recordings ran one physics tick per frame and are marked `"controls_only": 1` in `meta`.
Each segment gets its start state, boost pads included, and then only the cars' controls: no `drv`, `frac`,
`start_dt` or credits. Their segments also run through boost pad pickups and respawns. Their per-tick
arrays are in the game's car order (a segment's `perm` maps the arena's cars to it) and add `present`
(which cars and the ball are in the world), `pads` (each pad taken after the tick) and `boost`. The
scorers print a separate `[controls-only]` line for them: segments kept and dropped, and pickups, pad respawns,
demos and respawns on the game's tick.

`manifest.json` lists the recordings by split: `dev` and `test` (bot matches, 1v1 to 3v3), `bodies`
(every car body), `body_psyclops`, `camp` and `scn` (scripted scenarios with state sets), `h1` and `h2`
(held out until the others were exact), `uneven` (uneven teams, up to 8 cars) and `pads` (boost pad tests).

## Run

Build RocketGPU's Python module as its README says.

Dump the arena collision meshes from your copy of the game with
[RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper). They must match:

```bash
(cd /path/to/collision_meshes && sha256sum -c /path/to/this/meshes.sha256)
```

```bash
export PYTHONPATH=/path/to/RocketGPU/build/python ROCKETGPU_COLLISION_MESH_DIR=/path/to/collision_meshes
```

```bash
python score.py data --jobs 4 --json rocketgpu.json
```

`score_batch.py` takes the same arguments and writes the same JSON, stepping each recording's segments together.

`--splits dev,test` scores some splits. `--shard I/N` scores every Nth recording (one process per
GPU with `CUDA_VISIBLE_DEVICES`); `python score.py --merge all.json part*.json` adds the parts up.
Needs Python 3.9+ and numpy.

The full start state goes in through `Arena.set_full_state(state)`: a dict from field name
(`"car_states.pos_bt"`, `"car_car.points.normal"`, ...) to that field's bytes for each car, manifold or
point in turn. Fields are matched by name, so a build whose structs gained, lost or moved other fields
still loads the data; a field the build lacks, or whose size changed, is an error.
`Arena.get_full_state()` returns the same dict.

## RocketSim v3

```bash
cargo build --release --manifest-path v3/Cargo.toml
```

```bash
python score_v3.py data v3/target/release/fs_v3 --json v3.json
```

`ROCKETGPU_COLLISION_MESH_DIR` must be set. `score_v3.py` imports `score.py` but needs no RocketGPU
build. v3 is not deterministic from run to run: its segment pass moves in the last digit and its k30 and k120
errors in the fourth decimal.

## License

AGPL-3.0. See [LICENSE](LICENSE).
