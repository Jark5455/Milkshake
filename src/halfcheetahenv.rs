use crate::environment::{Environment, Spec, Trajectory};

use core::slice;
use libc::memcpy;
use libc::{c_char, c_int, c_void};
use mujoco_rs_sys::{
    mjData, mjModel, mjVFS, mj_defaultVFS, mj_deleteData, mj_deleteModel, mj_deleteVFS,
    mj_findFileVFS, mj_loadXML, mj_makeData, mj_makeEmptyFileVFS, mj_resetData,
};
use std::mem::MaybeUninit;

struct HalfCheetahEnv {
    pub model: *mut mjModel,
    pub data: *mut mjData,
    pub width: u32,
    pub height: u32,
    pub forward_reward_weight: f64,
    pub ctrl_cost_weight: f64,
    pub reset_noise_scale: f64,
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

    fn step(&mut self, _action: Vec<f64>) -> Box<dyn Trajectory> {
        todo!()
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
    ) -> Self {
        let halfcheetah_xml = include_str!("mujoco/halfcheetah.xml");
        let mut vfs: mjVFS;

        let width = width.unwrap_or(1920);
        let height = height.unwrap_or(1080);

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

            memcpy(
                vfs.filedata[file_idx as usize],
                halfcheetah_xml.as_ptr() as *const c_void,
                halfcheetah_xml.as_bytes().len(),
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

        HalfCheetahEnv {
            model,
            data,
            width,
            height,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
        }
    }

    pub fn control_cost(&self, action: Vec<f64>) -> f64 {
        self.ctrl_cost_weight * action.iter().map(|x| x.powi(2)).sum::<f64>()
    }

    pub fn reset(&mut self) {
        unsafe { mj_resetData(self.model, self.data) }
    }

    pub fn observation(&mut self) -> Vec<f64> {
        let pos = unsafe { (*self.data).qpos.clone() };
        let velocity = unsafe { (*self.data).qvel.clone() };

        let pos_vec = unsafe { slice::from_raw_parts(pos as *mut f64, 9).to_vec() };
        let velocity_vec = unsafe { slice::from_raw_parts(velocity as *mut f64, 9).to_vec() };

        [pos_vec, velocity_vec].concat()
    }
}
