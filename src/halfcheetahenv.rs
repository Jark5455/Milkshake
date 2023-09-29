use crate::environment::{Environment, Spec, Trajectory};
use libc::memcpy;
use libc::{c_char, c_int, c_void};
use mujoco_rs_sys::{
    mjModel, mjVFS, mj_defaultVFS, mj_deleteModel, mj_deleteVFS, mj_findFileVFS, mj_loadXML,
    mj_makeEmptyFileVFS,
};
use std::mem::MaybeUninit;

struct HalfCheetahEnv {
    pub model: *mut mjModel,
}

impl Environment for HalfCheetahEnv {
    fn action_spec(&self) -> Spec {
        todo!()
    }

    fn observation_spec(&self) -> Spec {
        todo!()
    }

    fn step(&mut self, _action: Vec<f64>) -> Box<dyn Trajectory> {
        todo!()
    }
}

impl HalfCheetahEnv {
    pub fn new() -> Self {
        let halfcheetah_xml = include_str!("mujoco/halfcheetah.xml");

        let mut vfs_uninit: MaybeUninit<mjVFS> = MaybeUninit::uninit();
        let vfs: &mut mjVFS;

        unsafe {
            let filename: *const c_char = "halfcheetah".as_ptr() as *const c_char;

            mj_defaultVFS(vfs_uninit.as_mut_ptr());
            vfs = vfs_uninit.assume_init_mut();

            mj_makeEmptyFileVFS(vfs, filename, halfcheetah_xml.as_bytes().len() as c_int);
            let file_idx = mj_findFileVFS(vfs, filename);

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
                vfs,
                err.as_mut_ptr(),
                err.len() as c_int,
            )
        };

        unsafe {
            mj_deleteVFS(vfs);
        }

        HalfCheetahEnv { model }
    }

    pub fn reset(&mut self) {}

    pub fn observation(&mut self) {}
}

impl Drop for HalfCheetahEnv {
    fn drop(&mut self) {
        unsafe {
            mj_deleteModel(self.model);
        }
    }
}
