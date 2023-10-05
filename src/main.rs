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
use std::fmt::Debug;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::copy_nonoverlapping;
use clap::Parser;
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
use tch::vision::image::save;
use crate::environment::{Environment, Terminate};
use crate::halfcheetahenv::HalfCheetahEnv;
use crate::replay_buffer::ReplayBuffer;
use crate::stockframe::StockFrame;
use crate::td3::TD3;

lazy_static! {
    static ref device: Arc<Device> = Arc::new(Device::cuda_if_available());
    static ref vs: Arc<nn::VarStore> = Arc::new(nn::VarStore::new(**device));
}

#[derive(Parser)]
struct Args {
    max_timesteps: Option<u32>,
    start_timesteps: Option<u32>,
    expl_noise: Option<f64>,
    eval_freq: Option<u32>,
    save_policy: Option<bool>
}

fn eval_td3(policy: &TD3, eval_episodes: Option<u32>) -> f64 {
    let eval_episodes = eval_episodes.unwrap_or(10);
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

fn run_td3(expl_noise: f64, max_timesteps: u32, start_timesteps: u32, eval_freq: u32, save_policy: bool) {
    let filename = "td3_halfcheetah";

    if !Path::new("./results").exists() {
        fs::create_dir_all("./results").expect("Failed to create results directory");
    }

    if !Path::new("./models").exists() {
        fs::create_dir_all("./models").expect("Failed to create models directory");
    }

    let mut env = HalfCheetahEnv::new(None, None, None, None, None, None, None);

    let state_dim = env.observation_spec().shape;
    let action_dim = env.action_spec().shape;
    let max_action = env.action_spec().max;

    let mut policy = TD3::new(state_dim as i64, action_dim as i64, max_action, None, None, None, None, None, None, None, None);
    let mut replaybuffer = ReplayBuffer::new(state_dim as i64, action_dim as i64, None);
    let mut evals = vec![eval_td3(&policy, None)];

    let mut ts = env.reset();
    let mut episode_reward = 0f64;
    let mut episode_timesteps = 0;
    let mut episode_num = 0;

    let mut rng = StdRng::from_entropy();
    let uniform = rand::distributions::Uniform::from(0f64..1f64);
    let normal = rand_distr::Normal::new(0f64, max_action * expl_noise).expect("Failed to make normal distribution");

    let mut action: Vec<f64>;
    for t in 0..max_timesteps {
        episode_timesteps += 1;

        if t < start_timesteps {
            action = (0..env.action_spec().shape).map(|_| uniform.sample(&mut rng)).collect();
        } else {
            action = policy.select_action(ts.observation().unwrap().iter().map(|act| act + normal.sample(&mut rng)).collect())
        }

        let next_ts = env.step(action.clone());
        let done =  ts.as_ref().as_any().downcast_ref::<Terminate>().is_some();
        let done_bool = if episode_timesteps < env.episode_length && done {1f64} else {0f64};
        replaybuffer.add(ts.observation().unwrap(), action, next_ts.observation().unwrap(), next_ts.reward().unwrap_or(0f64), done_bool);

        episode_reward += next_ts.reward().unwrap_or(0f64);
        ts = next_ts;

        if t >= start_timesteps {
            policy.train(&replaybuffer, None);
        }

        if done {
            println!("Total T: {} Episode Num: {} Episode T: {} Reward: {:.3}", t+1, episode_num+1, episode_timesteps, episode_reward);

            ts = env.step(Vec::new());
            episode_reward = 0f64;
            episode_timesteps = 0;
            episode_num += 1;
        }

        if (t + 1) % eval_freq == 0 {
            evals.push(eval_td3(&policy, None));
            let mut file = OpenOptions::new().write(true).open(format!("./results/{}.banan", filename)).expect(format!("Failed to open file ./results/{}.banan", filename).as_str());
            file.write(serde_json::to_string(&evals).expect("Failed to convert vals to string").as_bytes()).expect("Failed to write result");

            if save_policy {
                policy.save(format!("./results/{}_{}_steps.banan", filename, t)).expect("Failed to write policy to file");
            }
        }
    }
}

fn main() {
    let args = Args::parse();

    let expl_noise = args.expl_noise.unwrap_or(0.1);
    let max_timesteps = args.max_timesteps.unwrap_or(100000);
    let start_timesteps = args.start_timesteps.unwrap_or(25000);
    let eval_freq = args.eval_freq.unwrap_or(5000);
    let save_policy = args.save_policy.unwrap_or(true);

    run_td3(expl_noise, max_timesteps, start_timesteps, eval_freq, save_policy);
}
