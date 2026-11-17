//! fs_v3: RocketSim v3 worker for the full-state suite (score_v3.py drives it over stdin/stdout, little-endian f64 in,
//! f32 out). Per segment: n_cars, n_ticks, then per car [preset, team, CAR_F state floats], the ball [18 + asleep],
//! n_pads and per pad [x, y, cooldown]; then per tick, per car [8 controls, driving, credit gain]. Reply: per tick,
//! per car then the ball, pos (uu), vel (uu/s), ang vel, forward, right, up (18 f32). n_cars = -1 ends.
//!
//! State mode (score_v3_api.py), a segment starting with -2: n_cars, n_ticks, per car [preset, team], n_pads and
//! per pad [x, y]. The arena is kept for the segment and stepped API_WARM ticks first. Then per tick: per car [CAR_F
//! state floats (their controls are this tick's), 8 previous controls, seed], the ball [18 + asleep], and each
//! pad's cooldown. A car keeps its own state from tick to tick: only its body, boost, bump cooldown, last ball hit
//! and demolition are set from the sent floats, with the tick's controls. It is set to all of the sent state
//! ("seeded") on the first tick, where seed is not 0, and where its jump and dodge flags differ from the sent ones
//! (it did something else than the sender's car on the last tick). Every tick steps once and replies per car API_OUT
//! f32: the 18 body values, the state floats from has_jumped on (33), world contact [has, normal], controls,
//! previous controls, and the pad the car picked up this tick (-1 none); then the ball's 18.

use std::io::{self, BufWriter, Read, Write};

use glam::{Mat3A, Vec3A};
use rocketsim::{Arena, ArenaConfig, ArenaEvent, CarBodyConfig, CarControls, GameMode, Team};

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

/// Set a car's full state from CAR_F floats; `prev` is its previous controls (the state's own when None).
fn seed(arena: &mut Arena, id: usize, a: &[f64; CAR_F], prev: Option<CarControls>) {
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
    arena.edit_car_state_full(id, |s| {
        s.phys.pos = pos;
        s.phys.rot_mat = Mat3A::from_cols(fw, rt, up);
        s.phys.vel = vel;
        s.phys.ang_vel = ang;
        s.controls = controls;
        s.set_prev_controls(prev.unwrap_or(controls));
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

/// A car that keeps its own flags and timers: what the physics decides (body, boost, bump cooldown, last ball hit,
/// demolition) is set from CAR_F floats, with this tick's controls.
fn sync(arena: &mut Arena, id: usize, a: &[f64; CAR_F]) {
    let bytes = a.map(f64::to_le_bytes).concat();
    let mut r = In(&bytes[..]);
    let pos = r.v();
    let (fw, rt, up) = (r.v(), r.v(), r.v());
    let (vel, ang) = (r.v(), r.v());
    let controls = r.controls();
    for _ in 0..13 {
        r.f();
    }
    let boost = r.f() as f32;
    for _ in 0..9 {
        r.f();
    }
    let bump_cooldown_timer = r.f() as f32;
    let victim = r.f();
    let extra_hit = r.f();
    let is_demoed = r.b();
    let demo_respawn_timer = r.f() as f32;
    arena.edit_car_state_full(id, |s| {
        s.phys.pos = pos;
        s.phys.rot_mat = Mat3A::from_cols(fw, rt, up);
        s.phys.vel = vel;
        s.phys.ang_vel = ang;
        s.controls = controls;
        s.boost = boost;
        s.bump_cooldown_timer = bump_cooldown_timer;
        s.bump_cooldown_victim = (victim >= 0.0).then(|| victim as usize);
        s.last_extra_hit_tick = (extra_hit >= 0.0).then(|| extra_hit as u64);
        s.is_demoed = is_demoed;
        s.demo_respawn_timer = demo_respawn_timer;
    });
}

const API_WARM: usize = 2;

fn api<R: Read, W: Write>(inp: &mut In<R>, out: &mut W) {
    let mut buf: Vec<u8> = Vec::new();   // the reply goes out after the whole segment is read

    let (n, ticks) = (inp.f() as usize, inp.f() as usize);
    let mut arena = Arena::new_with_config(ArenaConfig::new(GameMode::Soccar));
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        let preset = PRESETS[inp.f() as usize];
        let team = if inp.f() == 0.0 { Team::Blue } else { Team::Orange };
        ids.push(arena.add_car(team, preset));
    }
    let pads: Vec<usize> = (0..inp.f() as usize)
        .map(|_| {
            let (x, y) = (inp.f() as f32, inp.f() as f32);
            (0..arena.num_boost_pads())
                .min_by(|&a, &b| {
                    let d = |i: usize| {
                        let p = arena.get_boost_pad_config(i).pos;
                        (p.x - x).powi(2) + (p.y - y).powi(2)
                    };
                    d(a).total_cmp(&d(b))
                })
                .unwrap()
        })
        .collect();
    for _ in 0..API_WARM {
        arena.step_tick();
    }
    let mut pickup = vec![-1f32; n];
    for t in 0..ticks {
        let mut next = Vec::with_capacity(n);
        for &id in &ids {
            let mut a = [0f64; CAR_F];
            for x in a.iter_mut() {
                *x = inp.f();
            }
            let prev = inp.controls();
            let seeded = inp.b();
            let acted = {
                let s = arena.get_car_state(id);
                let b = |i: usize| a[i] != 0.0;
                (s.has_jumped, s.has_double_jumped, s.has_flipped, s.is_flipping, s.is_jumping) != (b(26), b(27), b(28), b(34), b(35))
            };
            if t == 0 || seeded || acted {
                seed(&mut arena, id, &a, Some(prev));
            } else {
                sync(&mut arena, id, &a);
            }
            if !arena.get_car_state(id).is_demoed {
                arena.refresh_car_sticky_gate(id);
            }
            next.push(*arena.get_car_controls(id));
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
        } else {
            arena.wake_ball();
        }
        for &i in &pads {
            let cooldown = inp.f() as f32;
            arena.edit_boost_pad_state(i, |s| s.cooldown = cooldown);
        }
        for (k, &id) in ids.iter().enumerate() {
            arena.set_car_controls(id, next[k]);
        }
        pickup.fill(-1.0);
        for e in arena.step_tick() {
            if let ArenaEvent::CarPickupBoost(e) = e {
                pickup[e.car_idx] = pads.iter().position(|&i| i == e.boost_pad_idx).map_or(-2.0, |i| i as f32);
            }
        }
        let mut w = |x: f32| buf.extend_from_slice(&x.to_le_bytes());
        let b = |x: bool| x as u8 as f32;
        for (k, &id) in ids.iter().enumerate() {
            let s = arena.get_car_state(id);
            for v in [s.phys.pos, s.phys.vel, s.phys.ang_vel, s.phys.rot_mat.x_axis, s.phys.rot_mat.y_axis, s.phys.rot_mat.z_axis] {
                v.to_array().into_iter().for_each(&mut w);
            }
            [b(s.has_jumped), b(s.has_double_jumped), b(s.has_flipped)].into_iter().for_each(&mut w);
            s.flip_rel_torque.to_array().into_iter().for_each(&mut w);
            [s.jump_ticks as f32, s.flip_time, b(s.is_flipping), b(s.is_jumping), b(s.jump_hold_broken)].into_iter().for_each(&mut w);
            [s.air_time, s.air_time_since_jump, s.boost, s.handbrake_val, b(s.is_supersonic)].into_iter().for_each(&mut w);
            [s.supersonic_grace_timer, s.time_since_boosted, s.boosting_time, b(s.is_boosting), b(s.is_auto_flipping)].into_iter().for_each(&mut w);
            [s.auto_flip_timer, s.auto_flip_torque_scale, s.bump_cooldown_timer].into_iter().for_each(&mut w);
            w(s.bump_cooldown_victim.map_or(-1.0, |v| v as f32));
            w(s.last_extra_hit_tick.map_or(-1.0, |v| v as f32));
            [b(s.is_demoed), s.demo_respawn_timer, b(s.is_on_ground())].into_iter().for_each(&mut w);
            s.wheels_with_contact().iter().for_each(|x| w(b(x.is_some())));
            let wcn = s.world_contact_normal();
            w(b(wcn.is_some()));
            wcn.unwrap_or(Vec3A::ZERO).to_array().into_iter().for_each(&mut w);
            for c in [s.controls, *s.prev_controls()] {
                [c.throttle, c.steer, c.pitch, c.yaw, c.roll, b(c.jump), b(c.boost), b(c.handbrake)].into_iter().for_each(&mut w);
            }
            w(pickup[k]);
        }
        let p = &arena.get_ball_state().phys;
        for v in [p.pos, p.vel, p.ang_vel, p.rot_mat.x_axis, p.rot_mat.y_axis, p.rot_mat.z_axis] {
            v.to_array().into_iter().for_each(&mut w);
        }
    }
    out.write_all(&buf).unwrap();
    out.flush().unwrap();
}

fn main() {
    rocketsim::init(std::env::var("ROCKETGPU_COLLISION_MESH_DIR").expect("ROCKETGPU_COLLISION_MESH_DIR"), true).expect("meshes");
    let mut inp = In(io::stdin().lock());
    let mut out = BufWriter::new(io::stdout().lock());
    loop {
        let n = inp.f();
        if n == -2.0 {
            api(&mut inp, &mut out);
            continue;
        }
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
            seed(&mut arena, ids[k], a, None);
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
