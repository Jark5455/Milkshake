#![allow(nonstandard_style)]
#![allow(unused_imports)]
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
use crate::td3::TD3;

lazy_static! {
    static ref device: Arc<Device> = Arc::new(Device::cuda_if_available());
    static ref vs: Arc<nn::VarStore> = Arc::new(nn::VarStore::new(**device));
}

fn eval_td3(policy: TD3, eval_episodes: Option<u32>) -> f64 {
    let eval_episodes = eval_episodes.unwrap();
    let mut eval_env = HalfCheetahEnv::new(None, None, None, None, None, None, None);
    let mut ts = eval_env.reset();
    let mut avg_reward = 0f64;

    for _ in 0..eval_episodes {
        while ts.as_any().downcast_ref::<Terminate>().is_none() {
            let action = policy.select_action(ts.observation().unwrap());
            ts = eval_env.step(action);

            avg_reward += ts.reward().unwrap_or(0f64);
        }
    }

    avg_reward /= eval_episodes as f64;
    return avg_reward;
}

fn main() {

}
