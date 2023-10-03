use crate::environment::{Environment, Restart, Spec, Terminate, Trajectory, Transition};

use core::slice;
use libc::{c_char, c_int};
use mujoco_rs_sys::{
    mjData, mjModel, mjVFS, mj_defaultVFS, mj_deleteData, mj_deleteModel, mj_deleteVFS,
    mj_findFileVFS, mj_forward, mj_loadXML, mj_makeData, mj_makeEmptyFileVFS, mj_resetData,
    mj_step,
};
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::mem::MaybeUninit;
use std::ptr::copy_nonoverlapping;

struct HalfCheetahEnv {
    pub model: *mut mjModel,
    pub data: *mut mjData,
    pub width: u32,
    pub height: u32,
    pub frame_skip: u32,
    pub forward_reward_weight: f64,
    pub ctrl_cost_weight: f64,
    pub reset_noise_scale: f64,
    pub init_qpos: Vec<f64>,
    pub init_qvel: Vec<f64>,
    pub episode_length: u32,
    pub step: u32,
    pub episode_ended: bool,
}

impl Environment for HalfCheetahEnv {
    fn action_spec(&self) -> Spec {
        Spec {
            min: -1f64,
            max: 1f64,
            shape: 6,
        }
    }

    fn observation_spec(&self) -> Spec {
        Spec {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            shape: 18,
        }
    }

    fn step(&mut self, action: Vec<f64>) -> Box<dyn Trajectory> {
        if self.episode_ended {
            return self.reset();
        }

        self.step += 1;

        let x_position_before = unsafe { *(*self.data).qpos.offset(0) as f64 };
        self.do_simulation(action.clone());
        let x_pos_after = unsafe { *(*self.data).qpos.offset(0) as f64 };
        let x_velocity = unsafe {
            (x_pos_after - x_position_before)
                / ((*self.model).opt.timestep * self.frame_skip as f64)
        };

        let ctrl_cost = self.ctrl_cost_weight * self.control_cost(action.clone());
        let forward_reward = self.forward_reward_weight * x_velocity;

        let obs = self.observation();

        if self.step > self.episode_length {
            self.episode_ended = true;
            return Box::new(Terminate {
                observation: obs,
                reward: forward_reward - ctrl_cost,
            });
        }

        Box::new(Transition {
            observation: obs,
            reward: forward_reward - ctrl_cost,
        })
    }
}

impl Drop for HalfCheetahEnv {
    fn drop(&mut self) {
        unsafe {
            mj_deleteModel(self.model);
            mj_deleteData(self.data);
        }
    }
}

impl HalfCheetahEnv {
    pub fn new(
        forward_reward_weight: Option<f64>,
        ctrl_cost_weight: Option<f64>,
        reset_noise_scale: Option<f64>,
        width: Option<u32>,
        height: Option<u32>,
        frame_skip: Option<u32>,
        episode_length: Option<u32>,
    ) -> Self {
        let halfcheetah_xml = include_str!("mujoco/halfcheetah.xml");
        let mut vfs: mjVFS;

        let width = width.unwrap_or(1920);
        let height = height.unwrap_or(1080);
        let frame_skip = frame_skip.unwrap_or(5);
        let episode_length = episode_length.unwrap_or(1000);

        let forward_reward_weight = forward_reward_weight.unwrap_or(1f64);
        let ctrl_cost_weight = ctrl_cost_weight.unwrap_or(0.1);
        let reset_noise_scale = reset_noise_scale.unwrap_or(0.1);

        unsafe {
            let filename: *const c_char = "halfcheetah".as_ptr() as *const c_char;
            let mut vfs_uninit: MaybeUninit<mjVFS> = MaybeUninit::uninit();

            mj_defaultVFS(vfs_uninit.as_mut_ptr());
            vfs = vfs_uninit.assume_init();

            mj_makeEmptyFileVFS(
                &mut vfs,
                filename,
                halfcheetah_xml.as_bytes().len() as c_int,
            );
            let file_idx = mj_findFileVFS(&vfs, filename);

            copy_nonoverlapping(
                halfcheetah_xml.as_ptr(),
                vfs.filedata[file_idx as usize] as *mut u8,
                halfcheetah_xml.len(),
            );
        }

        let mut err = [0i8; 500];
        let model = unsafe {
            mj_loadXML(
                "halfcheetah.xml".as_ptr() as *const c_char,
                &vfs,
                err.as_mut_ptr(),
                err.len() as c_int,
            )
        };

        unsafe {
            mj_deleteVFS(&mut vfs);
        }

        let data = unsafe { mj_makeData(model) };

        let mut init_qpos = unsafe { vec![0f64; (*model).nq as usize] };
        let mut init_qvel = unsafe { vec![0f64; (*model).nv as usize] };

        unsafe {
            init_qpos.copy_from_slice(slice::from_raw_parts(
                (*data).qpos as *const f64,
                (*model).nq as usize,
            ));
            init_qvel.copy_from_slice(slice::from_raw_parts(
                (*data).qvel as *const f64,
                (*model).nv as usize,
            ));
        }

        HalfCheetahEnv {
            model,
            data,
            width,
            height,
            frame_skip,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            init_qpos,
            init_qvel,
            episode_length,
            step: 0,
            episode_ended: false,
        }
    }

    pub fn control_cost(&self, action: Vec<f64>) -> f64 {
        self.ctrl_cost_weight * action.iter().map(|x| x.powi(2)).sum::<f64>()
    }

    pub fn reset(&mut self) -> Box<dyn Trajectory> {
        unsafe { mj_resetData(self.model, self.data) }

        let noise_low = -self.reset_noise_scale;
        let noise_high = self.reset_noise_scale;

        let mut rng = StdRng::from_entropy();
        let uniform = rand::distributions::Uniform::from(noise_low..noise_high);
        let normal =
            rand_distr::Normal::new(0f64, 1f64).expect("Failed to make normal distribution");

        let qpos = unsafe {
            (0..(*self.model).nq)
                .map(|idx| self.init_qpos[idx as usize] + uniform.sample(&mut rng))
                .collect::<Vec<f64>>()
        };
        let qvel = unsafe {
            (0..(*self.model).nv)
                .map(|idx| self.init_qvel[idx as usize] + normal.sample(&mut rng))
                .collect::<Vec<f64>>()
        };

        unsafe {
            copy_nonoverlapping(qpos.as_ptr(), (*self.data).qpos, (*self.model).nq as usize);
            copy_nonoverlapping(qvel.as_ptr(), (*self.data).qvel, (*self.model).nv as usize);

            mj_forward(self.model, self.data);
        }

        self.step = 0;
        self.episode_ended = false;

        Box::new(Restart {
            observation: self.observation(),
        })
    }

    pub fn observation(&mut self) -> Vec<f64> {
        let mut pos = unsafe { vec![0f64; (*self.model).nq as usize] };
        let mut velocity = unsafe { vec![0f64; (*self.model).nv as usize] };

        unsafe {
            pos.copy_from_slice(slice::from_raw_parts(
                (*self.data).qpos as *const f64,
                (*self.model).nq as usize,
            ));
            velocity.copy_from_slice(slice::from_raw_parts(
                (*self.data).qvel as *const f64,
                (*self.model).nv as usize,
            ));
        }

        [pos, velocity].concat()
    }

    pub fn do_simulation(&mut self, ctrl: Vec<f64>) {
        assert_eq!(ctrl.len(), self.action_spec().shape as usize);
        unsafe { copy_nonoverlapping(ctrl.as_ptr(), (*self.data).ctrl, ctrl.len()) };

        for _ in 0..self.frame_skip {
            unsafe { mj_step(self.model, self.data) }
        }
    }
}
