extern crate rand;
extern crate rand_distr;

use crate::environment::{
    Environment, Mujoco, Restart, Spec, Terminate, Trajectory, Transition,
};

pub struct AntEnv {
    pub model: Box<crate::wrappers::mujoco::mjModel>,
    pub data: Box<crate::wrappers::mujoco::mjData>,
    pub width: u32,
    pub height: u32,
    pub frame_skip: u32,
    pub forward_reward_weight: f64,
    pub ctrl_cost_weight: f64,
    pub contact_cost_weight: f64,
    pub healthy_reward: f64,
    pub main_body: u32,
    pub terminate_when_unhealthy: bool,
    pub healthy_z_range: (f64, f64),
    pub contact_force_range: (f64, f64),
    pub reset_noise_scale: f64,
    pub init_qpos: Vec<f64>,
    pub init_qvel: Vec<f64>,
    pub episode_length: u32,
    pub step: u32,
    pub episode_ended: bool,
}

impl Environment for AntEnv {
    fn action_spec(&self) -> Spec {
        Spec {
            min: -1f64,
            max: 1f64,
            shape: 8,
        }
    }

    fn observation_spec(&self) -> Spec {
        Spec {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            shape: 107,
        }
    }

    fn step(&mut self, action: Vec<f64>) -> Box<dyn Trajectory> {
        if self.episode_ended {
            return self.reset();
        }

        self.step += 1;

        let x_position_before = unsafe { *self.data.xpos.offset(0) as f64 };
        self.do_simulation(action.clone());
        let x_pos_after = unsafe { *self.data.xpos.offset(0) as f64 };
        let x_velocity =
            (x_pos_after - x_position_before) / (self.model.opt.timestep as f64 * self.frame_skip as f64);

        let obs = self.observation();
        let reward = self.get_reward(x_velocity, action.clone());

        if self.step >= self.episode_length || (!self.is_healthy() && self.terminate_when_unhealthy){
            self.episode_ended = true;
            return Box::new(Terminate {
                observation: obs,
                reward: reward,
            });
        }

        Box::new(Transition {
            observation: obs,
            reward: reward,
        })
    }

    fn reset(&mut self) -> Box<dyn Trajectory> {

        unsafe { crate::wrappers::mujoco::mj_resetData(self.model.as_ref(), self.data.as_mut()) }

        let noise_low = -self.reset_noise_scale;
        let noise_high = self.reset_noise_scale;

        let mut rng = <rand::prelude::StdRng as rand::prelude::SeedableRng>::from_entropy();
        let uniform = rand::distributions::Uniform::from(noise_low..noise_high);
        let normal =
            rand_distr::Normal::new(0f64, 1f64).expect("Failed to make normal distribution");

        let qpos = (0..self.model.nq)
            .map(|idx| {
                self.init_qpos[idx as usize]
                    + rand::prelude::Distribution::sample(&uniform, &mut rng)
            })
            .collect::<Vec<f64>>();

        let qvel = (0..self.model.nv)
            .map(|idx| {
                self.init_qvel[idx as usize]
                    + rand::prelude::Distribution::sample(&normal, &mut rng)
            })
            .collect::<Vec<f64>>();

        if self.model.na == 0 {
            self.data.act = std::ptr::null_mut();
        }

        unsafe {
            std::ptr::copy_nonoverlapping(qpos.as_ptr(), self.data.qpos, self.model.nq as usize);
            std::ptr::copy_nonoverlapping(qvel.as_ptr(), self.data.qvel, self.model.nv as usize);

            crate::wrappers::mujoco::mj_forward(self.model.as_ref(), self.data.as_mut());
        }

        self.step = 0;
        self.episode_ended = false;

        Box::new(Restart {
            observation: self.observation(),
        })
    }
}

impl Mujoco for AntEnv {
    fn model(&mut self) -> &mut crate::wrappers::mujoco::mjModel {
        self.model.as_mut()
    }

    fn data(&mut self) -> &mut crate::wrappers::mujoco::mjData {
        self.data.as_mut()
    }

    fn observation(&self) -> Vec<f64> {
        let mut pos = vec![0f64; self.model.nq as usize];
        let mut velocity = vec![0f64; self.model.nv as usize];
        let mut contact_forces = vec![0f64; (self.model.nbody - 1) as usize * 6];

        unsafe {
            pos.copy_from_slice(core::slice::from_raw_parts(
                self.data.qpos as *const f64,
                self.model.nq as usize,
            ));
            velocity.copy_from_slice(core::slice::from_raw_parts(
                self.data.qvel as *const f64,
                self.model.nv as usize,
            ));
            contact_forces.copy_from_slice(core::slice::from_raw_parts(
                self.data.cfrc_ext as *const f64,
                (self.model.nbody - 1) as usize * 6,
            ));
        }

        [pos, velocity, contact_forces].concat()
    }
}

impl Drop for AntEnv {
    fn drop(&mut self) {
        unsafe {
            crate::wrappers::mujoco::mj_deleteModel(Box::leak(std::mem::take(&mut self.model)));
            crate::wrappers::mujoco::mj_deleteData(Box::leak(std::mem::take(&mut self.data)));
        }
    }
}

impl AntEnv {
    pub fn new(
        forward_reward_weight: Option<f64>,
        ctrl_cost_weight: Option<f64>,
        reset_noise_scale: Option<f64>,
        contact_cost_weight: Option<f64>,
        healthy_reward: Option<f64>,
        main_body: Option<u32>,
        terminate_when_unhealthy: Option<bool>,
        healthy_z_range: Option<(f64, f64)>,
        contact_force_range: Option<(f64, f64)>,
        width: Option<u32>,
        height: Option<u32>,
        frame_skip: Option<u32>,
        episode_length: Option<u32>,
    ) -> Self {
        let ant_xml = include_str!("../mujoco/ant.xml");
        let ant_file = "ant.xml".as_ptr() as *const libc::c_char;

        let width = width.unwrap_or(1920);
        let height = height.unwrap_or(1080);
        let frame_skip = frame_skip.unwrap_or(5);
        let episode_length = episode_length.unwrap_or(1000);

        let forward_reward_weight = forward_reward_weight.unwrap_or(1f64);
        let ctrl_cost_weight = ctrl_cost_weight.unwrap_or(0.5);
        let reset_noise_scale = reset_noise_scale.unwrap_or(0.1);

        let contact_cost_weight = contact_cost_weight.unwrap_or(0.0004f64);
        let healthy_reward = healthy_reward.unwrap_or(1f64);
        let main_body = main_body.unwrap_or(1);
        let terminate_when_unhealthy = terminate_when_unhealthy.unwrap_or(true);
        let healthy_z_range = healthy_z_range.unwrap_or((0.2f64, 1f64));
        let contact_force_range = contact_force_range.unwrap_or((-1f64, 1f64));

        unsafe {
            let layout = std::alloc::Layout::new::<crate::wrappers::mujoco::mjVFS>();
            let ptr = std::alloc::alloc(layout) as *mut crate::wrappers::mujoco::mjVFS;
            crate::wrappers::mujoco::mj_defaultVFS(ptr);

            let mut fs = Box::from_raw(ptr);

            let err = crate::wrappers::mujoco::mj_makeEmptyFileVFS(
                fs.as_mut(),
                ant_file,
                (ant_xml.len() + 1) as libc::c_int,
            );

            assert_eq!(err, 0);

            let file_idx = crate::wrappers::mujoco::mj_findFileVFS(fs.as_mut(), ant_file);

            assert!(file_idx >= 0);

            let fs_slice = std::slice::from_raw_parts_mut(
                fs.filedata[file_idx as usize] as *mut u8,
                ant_xml.len(),
            );
            fs_slice.copy_from_slice(ant_xml.as_bytes());

            let mut err = [0i8; 500];
            let model_raw = crate::wrappers::mujoco::mj_loadXML(
                ant_file,
                fs.as_ref(),
                err.as_mut_ptr(),
                err.len() as libc::c_int,
            );

            let model: Box<crate::wrappers::mujoco::mjModel> = Box::from_raw(model_raw);
            let data: Box<crate::wrappers::mujoco::mjData> =
                Box::from_raw(crate::wrappers::mujoco::mj_makeData(model.as_ref()));

            let mut init_qpos = vec![0f64; model.nq as usize];
            let mut init_qvel = vec![0f64; model.nv as usize];

            init_qpos.copy_from_slice(core::slice::from_raw_parts(
                data.qpos as *const f64,
                model.nq as usize,
            ));

            init_qvel.copy_from_slice(core::slice::from_raw_parts(
                data.qvel as *const f64,
                model.nv as usize,
            ));

            crate::wrappers::mujoco::mj_deleteVFS(fs.as_mut());

            AntEnv {
                model,
                data,
                width,
                height,
                frame_skip,
                forward_reward_weight,
                ctrl_cost_weight,
                contact_cost_weight,
                healthy_reward,
                main_body,
                terminate_when_unhealthy,
                healthy_z_range,
                contact_force_range,
                reset_noise_scale,
                init_qpos,
                init_qvel,
                episode_length,
                step: 0,
                episode_ended: true,
            }
        }
    }

    pub fn health_reward(&self) -> f64 {
        match self.is_healthy() {
            true => self.healthy_reward,
            false => 0f64
        }
    }

    pub fn control_cost(&self, action: Vec<f64>) -> f64 {
        self.ctrl_cost_weight * action.iter().map(|x| x.powi(2)).sum::<f64>()
    }

    pub fn contact_cost(&self) -> f64 {
        let contact_forces = self.contact_forces();
        self.contact_cost_weight * contact_forces.iter().map(|x| x.powi(2)).sum::<f64>()
    }

    pub fn contact_forces(&self) -> Vec<f64> {
        let contact_forces = unsafe { core::slice::from_raw_parts(self.data.cfrc_ext as *const f64, (self.model.nbody - 1) as usize * 6) };
        let mut contact_forces = contact_forces.to_vec();

        for idx in 0..contact_forces.len() {
            if contact_forces[idx] < self.contact_force_range.0 {
                contact_forces[idx] = self.contact_force_range.0;
            } else if contact_forces[idx] > self.contact_force_range.1 {
                contact_forces[idx] = self.contact_force_range.1;
            }
        }

        contact_forces
    }

    pub fn is_healthy(&self) -> bool {
        let qpos = unsafe { std::slice::from_raw_parts(self.data.qpos as *const f64, self.model.nq as usize) };
        let qvel = unsafe { std::slice::from_raw_parts(self.data.qvel as *const f64, self.model.nv as usize) };

        let state = [qpos, qvel].concat();

        for val in &state {
            if !val.is_finite() {
                return false;
            }
        }

        state[2] <= self.healthy_z_range.1 && state[2] >= self.healthy_z_range.0
    }

    pub fn get_reward(&self, x_velocity: f64, action: Vec<f64>) -> f64 {
        let forward_reward = self.forward_reward_weight * x_velocity;
        let healthy_reward = self.health_reward();
        let rewards = forward_reward + healthy_reward;

        let ctrl_cost = self.control_cost(action.clone());
        let contact_cost = self.contact_cost();
        let costs = ctrl_cost + contact_cost;

        rewards - costs
    }

    pub fn do_simulation(&mut self, ctrl: Vec<f64>) {
        assert_eq!(ctrl.len(), self.action_spec().shape as usize);
        unsafe { std::ptr::copy_nonoverlapping(ctrl.as_ptr(), self.data.ctrl, ctrl.len()) };

        for _ in 0..self.frame_skip {
            unsafe { crate::wrappers::mujoco::mj_step(self.model.as_ref(), self.data.as_mut()) }
        }

        unsafe { crate::wrappers::mujoco::mj_rnePostConstraint(self.model.as_ref(), self.data.as_mut()) }
    }
}
