# RocketGPU full-state suite

121 Rocket League recordings with the game's full physics state, and the scripts that score
[RocketGPU](https://github.com/thecosmicskye/RocketGPU) and RocketSim v3 against them.

Each segment starts from the game's state at one tick and runs open loop for up to 120 ticks on the
game's recorded inputs. Each tick, every car and the ball are compared with the game:

- **exact**: position, velocity, angular velocity and rotation all equal to the game's, bit for bit.
- **segment pass**: car-ticks whose normalized error is below 1 (position / 2 uu, velocity / 3 uu/s,
  angular velocity / 0.05 rad/s, forward and up axes / 0.05, over the car and the ball).
- **mean error**: that error, capped at 10, at 1, 30 and 120 ticks.

## Results

12,038,088 body-ticks in 22,557 segments.

| | RocketGPU (CUDA) | RocketSim v3 0.3.5 |
|---|---|---|
| Exact body-ticks | 100.0000% | 4.4884% |
| Exact segments | 100.000% | 0.000% |
| Segment pass | 100.000% | 85.930% |
| Mean error k1 / k30 / k120 | 0 / 0 / 0 | 0.80 / 0.64 / 1.65 |

RocketGPU starts from the full state (bodies, broadphase, contact manifolds, timers). RocketSim v3
starts from the public state (what its `CarState` and `BallState` hold).

## Data

Recorded in Rocket League. The truth is the game's own state at each tick. Each segment's start state is the
game's state at that tick in RocketGPU's state fields (the names `Arena.get_full_state()` uses), so it loads
with `set_full_state` in
[RocketGPU 73c3e9f7](https://github.com/thecosmicskye/RocketGPU/commit/73c3e9f72d24e79968611d52a9ed4837d30b23c4)
or later builds with the same fields.

The recordings are a release asset (744 MB):

```bash
gh release download v1 -R thecosmicskye/RocketGPU-fullstate-suite
```

```bash
tar xf fullstate-suite-v1.tar && sha256sum -c data.sha256
```

`data/NAME.npz` holds, per recording:

| Array | |
|---|---|
| `frame`, `ctl`, `drv`, `frac`, `start_dt`, `held` | per tick: game tick, each car's controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake), whether the game lets it drive, the render frame's interpolation fraction and start time, all cars held (not scored) |
| `truth` | per tick, each car then the ball: position and velocity (Bullet units, uu / 50), angular velocity, rotation matrix by rows |
| `credit_tick`, `credit_slot`, `credit_gain` | the game's boost pad pickups |
| `s.*`, `meta` | per segment, the full start state (`Arena.get_full_state()` fields), and the cars, boost pads and mutators |
| `public` | per segment, the public start state (the rows `score_v3.py` sends) |

`manifest.json` lists the recordings by split: `dev` and `test` (bot matches, 1v1 to 3v3), `bodies`
(every car body), `body_psyclops`, `camp` and `scn` (scripted scenarios with state sets), `h1`
(held out until the others were exact).

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
