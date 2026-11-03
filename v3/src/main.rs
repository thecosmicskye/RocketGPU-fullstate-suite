//! fs_v3: RocketSim v3 worker for the full-state suite (score_v3.py drives it over stdin/stdout, little-endian f64 in,
//! f32 out). Per segment: n_cars, n_ticks, then per car [preset, team, CAR_F state floats], the ball [18 + asleep],
//! n_pads and per pad [x, y, cooldown]; then per tick, per car [8 controls, driving, credit gain]. Reply: per tick,
//! per car then the ball, pos (uu), vel (uu/s), ang vel, forward, right, up (18 f32). n_cars = -1 ends.

use std::io::{self, BufWriter, Read, Write};

use glam::{Mat3A, Vec3A};
use rocketsim::{Arena, ArenaConfig, CarBodyConfig, CarControls, GameMode, Team};

const CAR_F: usize = 59;
const PRESETS: [CarBodyConfig; 7] = [
    CarBodyConfig::OCTANE,
    CarBodyConfig::DOMINUS,
    CarBodyConfig::PLANK,
    CarBodyConfig::BREAKOUT,
    CarBodyConfig::HYBRID,
    CarBodyConfig::MERC,
    CarBodyConfig::PSYCLOPS,
];

struct In<R: Read>(R);
impl<R: Read> In<R> {
    fn f(&mut self) -> f64 {
        let mut b = [0u8; 8];
        self.0.read_exact(&mut b).expect("stdin");
        f64::from_le_bytes(b)
    }
    fn v(&mut self) -> Vec3A {
        Vec3A::new(self.f() as f32, self.f() as f32, self.f() as f32)
    }
    fn b(&mut self) -> bool {
        self.f() != 0.0
    }
    fn controls(&mut self) -> CarControls {
        CarControls {
            throttle: self.f() as f32,
            steer: self.f() as f32,
            pitch: self.f() as f32,
            yaw: self.f() as f32,
            roll: self.f() as f32,
            jump: self.b(),
            boost: self.b(),
            handbrake: self.b(),
        }
    }
}

fn main() {
    rocketsim::init(std::env::var("ROCKETGPU_COLLISION_MESH_DIR").expect("ROCKETGPU_COLLISION_MESH_DIR"), true).expect("meshes");
    let mut inp = In(io::stdin().lock());
    let mut out = BufWriter::new(io::stdout().lock());
    loop {
        let n = inp.f();
        if n < 0.0 {
            break;
        }
        let (n, ticks) = (n as usize, inp.f() as usize);
        let mut arena = Arena::new_with_config(ArenaConfig::new(GameMode::Soccar));
        let mut ids = Vec::with_capacity(n);
        let mut states = Vec::with_capacity(n);
        for _ in 0..n {
            let preset = PRESETS[inp.f() as usize];
            let team = if inp.f() == 0.0 { Team::Blue } else { Team::Orange };
            ids.push(arena.add_car(team, preset));
            let mut a = [0f64; CAR_F];
            for x in a.iter_mut() {
                *x = inp.f();
            }
            states.push(a);
        }
        for (k, a) in states.iter().enumerate() {
            let bytes = a.map(f64::to_le_bytes).concat();
            let mut r = In(&bytes[..]);
            let pos = r.v();
            let (fw, rt, up) = (r.v(), r.v(), r.v());
            let (vel, ang) = (r.v(), r.v());
            let controls = r.controls();
            let (hj, hdj, hf) = (r.b(), r.b(), r.b());
            let flip_rel_torque = r.v();
            let jump_ticks = r.f() as u32;
            let flip_time = r.f() as f32;
            let (is_flipping, is_jumping, jump_hold_broken) = (r.b(), r.b(), r.b());
            let (air_time, air_time_since_jump, boost, handbrake_val) =
                (r.f() as f32, r.f() as f32, r.f() as f32, r.f() as f32);
            let is_supersonic = r.b();
            let (supersonic_grace_timer, time_since_boosted, boosting_time) = (r.f() as f32, r.f() as f32, r.f() as f32);
            let (is_boosting, is_auto_flipping) = (r.b(), r.b());
            let (auto_flip_timer, auto_flip_torque_scale, bump_cooldown_timer) = (r.f() as f32, r.f() as f32, r.f() as f32);
            let victim = r.f();
            let extra_hit = r.f();
            let is_demoed = r.b();
            let demo_respawn_timer = r.f() as f32;
            let on_ground = r.b();
            let wheels = [r.b(), r.b(), r.b(), r.b()];
            arena.edit_car_state_full(ids[k], |s| {
                s.phys.pos = pos;
                s.phys.rot_mat = Mat3A::from_cols(fw, rt, up);
                s.phys.vel = vel;
                s.phys.ang_vel = ang;
                s.controls = controls;
                s.set_prev_controls(controls);
                s.set_contact_flags(on_ground, wheels);
                s.has_jumped = hj;
                s.has_double_jumped = hdj;
                s.has_flipped = hf;
                s.flip_rel_torque = flip_rel_torque;
                s.jump_ticks = jump_ticks;
                s.flip_time = flip_time;
                s.is_flipping = is_flipping;
                s.is_jumping = is_jumping;
                s.jump_hold_broken = jump_hold_broken;
                s.air_time = air_time;
                s.air_time_since_jump = air_time_since_jump;
                s.boost = boost;
                s.handbrake_val = handbrake_val;
                s.is_supersonic = is_supersonic;
                s.supersonic_grace_timer = supersonic_grace_timer;
                s.time_since_boosted = time_since_boosted;
                s.boosting_time = boosting_time;
                s.is_boosting = is_boosting;
                s.is_auto_flipping = is_auto_flipping;
                s.auto_flip_timer = auto_flip_timer;
                s.auto_flip_torque_scale = auto_flip_torque_scale;
                s.bump_cooldown_timer = bump_cooldown_timer;
                s.bump_cooldown_victim = (victim >= 0.0).then(|| victim as usize);
                s.last_extra_hit_tick = (extra_hit >= 0.0).then(|| extra_hit as u64);
                s.is_demoed = is_demoed;
                s.demo_respawn_timer = demo_respawn_timer;
            });
        }
        let pos = inp.v();
        let (fw, rt, up) = (inp.v(), inp.v(), inp.v());
        let (vel, ang) = (inp.v(), inp.v());
        let asleep = inp.b();
        arena.edit_ball_state(|b| {
            b.phys.pos = pos;
            b.phys.rot_mat = Mat3A::from_cols(fw, rt, up);
            b.phys.vel = vel;
            b.phys.ang_vel = ang;
        });
        if asleep {
            arena.sleep_ball();
        }
        let pads = inp.f() as usize;
        for _ in 0..pads {
            let (x, y, cooldown) = (inp.f() as f32, inp.f() as f32, inp.f() as f32);
            let i = (0..arena.num_boost_pads())
                .min_by(|&a, &b| {
                    let d = |i: usize| {
                        let p = arena.get_boost_pad_config(i).pos;
                        (p.x - x).powi(2) + (p.y - y).powi(2)
                    };
                    d(a).total_cmp(&d(b))
                })
                .unwrap();
            arena.edit_boost_pad_state(i, |s| s.cooldown = cooldown);
        }
        for _ in 0..ticks {
            let mut credits = Vec::new();
            for &id in &ids {
                let mut c = inp.controls();
                if !inp.b() {
                    c = CarControls { steer: c.steer, ..Default::default() };
                }
                arena.set_car_controls(id, c);
                credits.push(inp.f() as f32);
            }
            arena.step_tick();
            for (k, &id) in ids.iter().enumerate() {
                let gain = credits[k];
                if gain > 0.0 && !arena.get_car_state(id).is_demoed {
                    let b = arena.get_car_state(id).boost;
                    let m = (gain / 0.12).round();
                    let nb = if (1.0..=4.0).contains(&m) && (gain - 0.12 * m).abs() < 1e-3 { (b + 12.0 * m).min(100.0) } else { 100.0 };
                    arena.set_car_boost(id, nb);
                }
            }
            let mut put = |p: &rocketsim::PhysState| {
                for v in [p.pos, p.vel, p.ang_vel, p.rot_mat.x_axis, p.rot_mat.y_axis, p.rot_mat.z_axis] {
                    for x in v.to_array() {
                        out.write_all(&x.to_le_bytes()).unwrap();
                    }
                }
            };
            for &id in &ids {
                put(&arena.get_car_state(id).phys);
            }
            put(&arena.get_ball_state().phys);
        }
        out.flush().unwrap();
    }
}
