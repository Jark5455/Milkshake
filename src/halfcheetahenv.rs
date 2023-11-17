extern crate rand;
extern crate rand_distr;

use crate::environment::{
    Environment, Mujoco, MujocoEnvironment, Restart, Spec, Terminate, Trajectory, Transition,
};

pub struct HalfCheetahEnv {
    pub model: Box<crate::mujoco::mjModel>,
    pub data: Box<crate::mujoco::mjData>,
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

        let x_position_before = unsafe { *self.data.qpos.offset(0) as f64 };
        self.do_simulation(action.clone());
        let x_pos_after = unsafe { *self.data.qpos.offset(0) as f64 };
        let x_velocity =
            (x_pos_after - x_position_before) / (self.model.opt.timestep * self.frame_skip as f64);

        let ctrl_cost = self.ctrl_cost_weight * self.control_cost(action.clone());
        let forward_reward = self.forward_reward_weight * x_velocity;

        let obs = self.observation();

        if self.step >= self.episode_length {
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

    fn reset(&mut self) -> Box<dyn Trajectory> {
        unsafe { crate::mujoco::mj_resetData(self.model.as_ref(), self.data.as_mut()) }

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

        unsafe {
            std::ptr::copy_nonoverlapping(qpos.as_ptr(), self.data.qpos, self.model.nq as usize);
            std::ptr::copy_nonoverlapping(qvel.as_ptr(), self.data.qvel, self.model.nv as usize);

            crate::mujoco::mj_forward(self.model.as_ref(), self.data.as_mut());
        }

        self.step = 0;
        self.episode_ended = false;

        Box::new(Restart {
            observation: self.observation(),
        })
    }
}

impl Mujoco for HalfCheetahEnv {
    fn model(&mut self) -> &mut crate::mujoco::mjModel {
        self.model.as_mut()
    }

    fn data(&mut self) -> &mut crate::mujoco::mjData {
        self.data.as_mut()
    }

    fn observation(&self) -> Vec<f64> {
        let mut pos = vec![0f64; self.model.nq as usize];
        let mut velocity = vec![0f64; self.model.nv as usize];

        unsafe {
            pos.copy_from_slice(core::slice::from_raw_parts(
                self.data.qpos as *const f64,
                self.model.nq as usize,
            ));
            velocity.copy_from_slice(core::slice::from_raw_parts(
                self.data.qvel as *const f64,
                self.model.nv as usize,
            ));
        }

        [pos, velocity].concat()
    }
}

impl MujocoEnvironment for HalfCheetahEnv {}

impl Drop for HalfCheetahEnv {
    fn drop(&mut self) {
        unsafe {
            crate::mujoco::mj_deleteModel(Box::leak(std::mem::replace(
                &mut self.model,
                Box::new(crate::mujoco::mjModel::default()),
            )));
            crate::mujoco::mj_deleteData(Box::leak(std::mem::replace(
                &mut self.data,
                Box::new(crate::mujoco::mjData::default()),
            )));
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
        let halfcheetah_file = "halfcheetah.xml".as_ptr() as *const libc::c_char;

        let width = width.unwrap_or(1920);
        let height = height.unwrap_or(1080);
        let frame_skip = frame_skip.unwrap_or(5);
        let episode_length = episode_length.unwrap_or(1000);

        let forward_reward_weight = forward_reward_weight.unwrap_or(1f64);
        let ctrl_cost_weight = ctrl_cost_weight.unwrap_or(0.1);
        let reset_noise_scale = reset_noise_scale.unwrap_or(0.1);

        unsafe {
            let mut fs = Box::new(crate::mujoco::mjVFS::default());
            crate::mujoco::mj_defaultVFS(std::ops::DerefMut::deref_mut(&mut fs));

            let err = crate::mujoco::mj_makeEmptyFileVFS(
                fs.as_mut(),
                halfcheetah_file,
                (halfcheetah_xml.len() + 1) as libc::c_int,
            );

            assert_eq!(err, 0);

            let file_idx = crate::mujoco::mj_findFileVFS(fs.as_mut(), halfcheetah_file);

            assert!(file_idx >= 0);

            let fs_slice = std::slice::from_raw_parts_mut(
                fs.filedata[file_idx as usize] as *mut u8,
                halfcheetah_xml.len(),
            );
            fs_slice.copy_from_slice(halfcheetah_xml.as_bytes());

            let mut err = [0i8; 500];
            let model_raw = crate::mujoco::mj_loadXML(
                halfcheetah_file,
                fs.as_ref(),
                err.as_mut_ptr(),
                err.len() as libc::c_int,
            );

            let model: Box<crate::mujoco::mjModel> = Box::from_raw(model_raw);
            let data: Box<crate::mujoco::mjData> = Box::from_raw(crate::mujoco::mj_makeData(model.as_ref()));

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

            crate::mujoco::mj_deleteVFS(fs.as_mut());

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
                episode_ended: true,
            }
        }
    }

    pub fn control_cost(&self, action: Vec<f64>) -> f64 {
        self.ctrl_cost_weight * action.iter().map(|x| x.powi(2)).sum::<f64>()
    }

    pub fn do_simulation(&mut self, ctrl: Vec<f64>) {
        assert_eq!(ctrl.len(), self.action_spec().shape as usize);
        unsafe { std::ptr::copy_nonoverlapping(ctrl.as_ptr(), self.data.ctrl, ctrl.len()) };

        for _ in 0..self.frame_skip {
            unsafe { crate::mujoco::mj_step(self.model.as_ref(), self.data.as_mut()) }
        }
    }
}
