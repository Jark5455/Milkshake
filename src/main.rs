#![allow(nonstandard_style)]
#![allow(dead_code)]

extern crate core;

mod environment;
mod halfcheetahenv;
mod replay_buffer;
mod stockenv;
mod stockframe;
mod td3;
mod tests;

use std::any::{Any, TypeId};
use std::mem::MaybeUninit;
use std::ptr::copy_nonoverlapping;
use crate::stockframe::StockFrame;
use dotenv::dotenv;
use lazy_static::lazy_static;
use polars::export::chrono::{Duration, Utc};
use polars::prelude::FillNullStrategy;
use std::sync::Arc;
use libc::{c_char, c_int};
use mujoco_rs_sys::{mj_defaultVFS, mj_findFileVFS, mj_makeEmptyFileVFS, mjVFS};
use rand::prelude::{Distribution, StdRng};
use rand::SeedableRng;
use tch::nn;
use tch::Device;
use crate::environment::{Environment, Terminate};
use crate::halfcheetahenv::HalfCheetahEnv;

lazy_static! {
    static ref device: Arc<Device> = Arc::new(Device::cuda_if_available());
    static ref vs: Arc<nn::VarStore> = Arc::new(nn::VarStore::new(**device));
}

fn main() {
    let mut env = HalfCheetahEnv::new(None, None, None, None, None, None, None);
    let mut rng = StdRng::from_entropy();
    let uniform = rand::distributions::Uniform::from(0f64..1f64);

    let mut iter = 0;
    while iter < 5 {
        let ts = env.step((0..env.action_spec().shape).map(|idx| uniform.sample(&mut rng)).collect());

        println!("step: {}, obs: {:?}, reward: {:?}", env.step, ts.observation(), ts.reward());

        if ts.as_ref().as_any().downcast_ref::<Terminate>().is_some() {
            println!("episode ended: {}", iter);
            iter += 1
        }
    }
}
