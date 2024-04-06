#![allow(nonstandard_style)]
#![allow(dead_code)]

mod environment;
mod optimizer;
mod replay_buffer;
mod stockframe;
mod td3;
mod tests;
mod viewer;
mod wrappers;

use crate::environment::{Environment, Mujoco, Terminate};
use crate::environment::halfcheetahenv::HalfCheetahEnv;
use crate::environment::antenv::AntEnv;
use crate::environment::stockenv::StockEnv;
use crate::replay_buffer::ReplayBuffer;

use crate::td3::TD3;
use crate::viewer::Viewer;

lazy_static::lazy_static! {
    static ref device: std::sync::Arc<tch::Device> = std::sync::Arc::new(tch::Device::cuda_if_available());
}

#[derive(clap::Parser)]
struct Args {
    #[arg(long)]
    env: String,
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    Train {
        #[arg(long)]
        actor_opt: String,
        #[arg(long)]
        critic_opt: String,
        #[arg(long)]
        max_timesteps: Option<u32>,
        #[arg(long)]
        start_timesteps: Option<u32>,
        #[arg(long)]
        expl_noise: Option<f64>,
        #[arg(long)]
        eval_freq: Option<u32>,
        #[arg(long)]
        save_policy: Option<bool>,
    },

    Run {
        #[arg(long)]
        savefile: String,
    },
}

fn eval_td3(policy: &TD3, env: &mut Box<dyn Environment>, eval_episodes: Option<u32>) -> f64 {
    let eval_episodes = eval_episodes.unwrap_or(10);
    let mut ts = env.reset();
    let mut avg_reward = 0f64;

    for _ in 0..eval_episodes {
        while ts.as_any().downcast_ref::<Terminate>().is_none() {
            let action = policy.select_action(ts.observation());
            ts = env.step(action);

            avg_reward += ts.reward().unwrap_or(0f64);
        }
    }

    avg_reward /= eval_episodes as f64;
    avg_reward
}

fn run_td3(
    env: &str,
    filename: &str,
    expl_noise: f64,
    max_timesteps: u32,
    start_timesteps: u32,
    eval_freq: u32,
    save_policy: bool,
    actor_opt: &str,
    critic_opt: &str,
) {
    if !std::path::Path::new("./results").exists() {
        std::fs::create_dir_all("./results").expect("Failed to create results directory");
    }

    if !std::path::Path::new("./models").exists() {
        std::fs::create_dir_all("./models").expect("Failed to create models directory");
    }

    let envs: (Box<dyn Environment>, Box<dyn Environment>) = match env {
        "halfcheetah" => {
            let train_env = Box::new(HalfCheetahEnv::new(
                None, None, None, None, None, None, None,
            ));

            let eval_env = Box::new(HalfCheetahEnv::new(
                None, None, None, None, None, None, None,
            ));

            (train_env, eval_env)
        }

        "ant" => {
            let train_env = Box::new(AntEnv::new(
                None, None, None, None, None, None, None, None, None, None, None, None, None,
            ));

            let eval_env = Box::new(AntEnv::new(
                None, None, None, None, None, None, None, None, None, None, None, None, None,
            ));

            (train_env, eval_env)
        }

        "stockenv" => {
            let end = polars::export::chrono::Utc::now()
                .date_naive()
                .and_hms_micro_opt(0, 0, 0, 0)
                .unwrap();
            let start = end - polars::export::chrono::Duration::days(15);

            let train_env = Box::new(StockEnv::new(start, end));
            let eval_env = train_env.clone();

            (train_env, eval_env)
        }

        &_ => {
            panic!("Invalid Environment Selection")
        }
    };

    let mut train_env = envs.0;
    let mut eval_env = envs.1;

    let state_dim = train_env.observation_spec().shape;
    let action_dim = train_env.action_spec().shape;
    let max_action = train_env.action_spec().max;

    let mut policy = TD3::new(
        state_dim as i64,
        action_dim as i64,
        max_action,
        actor_opt,
        critic_opt,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("Failed to create TD3 Policy");

    let mut replaybuffer = ReplayBuffer::new(state_dim as i64, action_dim as i64, None);
    let mut evals = vec![eval_td3(&policy, &mut eval_env, None)];

    let mut ts = train_env.reset();
    let mut episode_reward = 0f64;
    let mut episode_timesteps = 0;
    let mut episode_num = 0;

    let mut rng = <rand::prelude::StdRng as rand::prelude::SeedableRng>::from_entropy();
    let uniform = rand::distributions::Uniform::from(0f64..1f64);
    let normal = rand_distr::Normal::new(0f64, max_action * expl_noise)
        .expect("Failed to make normal distribution");

    let mut action: Vec<f64>;
    for t in 0..max_timesteps {
        episode_timesteps += 1;

        if t < start_timesteps {
            action = (0..train_env.action_spec().shape)
                .map(|_| rand::prelude::Distribution::sample(&uniform, &mut rng))
                .collect();
        } else {
            action = policy.select_action(
                ts.observation()
                    .iter()
                    .map(|act| act + rand::prelude::Distribution::sample(&normal, &mut rng))
                    .collect(),
            )
        }

        let next_ts = train_env.step(action.clone());
        let done = next_ts
            .as_ref()
            .as_any()
            .downcast_ref::<Terminate>()
            .is_some();

        let done_bool = match done {
            true => 1f64,
            false => 0f64,
        };

        replaybuffer.add(
            ts.observation(),
            action,
            next_ts.observation(),
            next_ts.reward().unwrap_or(0f64),
            done_bool,
        );

        episode_reward += next_ts.reward().unwrap_or(0f64);
        ts = next_ts;

        if t >= start_timesteps {
            policy.train(&replaybuffer, None);
        }

        if done {
            println!(
                "Total T: {} Episode Num: {} Episode T: {} Reward: {:.3}",
                t + 1,
                episode_num + 1,
                episode_timesteps,
                episode_reward
            );

            ts = train_env.step(Vec::new());
            episode_reward = 0f64;
            episode_timesteps = 0;
            episode_num += 1;
        }

        if (t + 1) % eval_freq == 0 {
            evals.push(eval_td3(&policy, &mut eval_env, None));
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(format!("./results/{}.banan", filename))
                .unwrap_or_else(|_| panic!("Failed to open file ./results/{}.banan", filename));

            std::io::Write::write_all(
                &mut file,
                serde_json::to_string_pretty(&evals)
                    .expect("Failed to convert vals to string")
                    .as_bytes(),
            )
            .expect("Failed to write result");

            if save_policy {
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(format!("./models/{}_{}_steps.banan", filename, t + 1))
                    .expect("Failed to open file to save model");

                std::io::Write::write_all(
                    &mut file,
                    serde_json::to_string_pretty(&policy)
                        .expect("Failed to serialize td3 to json")
                        .as_bytes(),
                )
                .expect("Failed to write td3 to file");
            }
        }
    }
}

fn load_td3(filename: String) -> TD3 {
    let data = std::fs::read_to_string(filename.clone())
        .unwrap_or_else(|_| panic!("Failed to read file: {}", filename.clone()));
    serde_json::from_str(data.as_str())
        .unwrap_or_else(|_| panic!("Failed to parse td3 from file: {}", filename.clone()))
}

fn main() {
    println!("Cuda Enabled: {}", device.is_cuda());

    let args = <Args as clap::Parser>::parse();

    match args.command {
        Commands::Train {
            actor_opt,
            critic_opt,
            expl_noise,
            max_timesteps,
            start_timesteps,
            eval_freq,
            save_policy,
        } => {
            let expl_noise = expl_noise.unwrap_or(0.1);
            let max_timesteps = max_timesteps.unwrap_or(100000);
            let start_timesteps = start_timesteps.unwrap_or(25000);
            let eval_freq = eval_freq.unwrap_or(5000);
            let save_policy = save_policy.unwrap_or(false);

            let filename = format!(
                "td3_{}_{}_{}",
                args.env,
                actor_opt.to_lowercase(),
                critic_opt.to_lowercase()
            );

            run_td3(
                args.env.as_str(),
                filename.as_str(),
                expl_noise,
                max_timesteps,
                start_timesteps,
                eval_freq,
                save_policy,
                actor_opt.as_str(),
                critic_opt.as_str(),
            );
        }

        Commands::Run { savefile } => {
            let td3 = load_td3(savefile);

            let env: Box<dyn Mujoco> = match args.env.as_str() {
                "halfcheetah" => Box::new(HalfCheetahEnv::new(
                    None, None, None, None, None, None, None,
                )),

                "ant" => Box::new(AntEnv::new(
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                )),

                &_ => {
                    panic!("Selected Environment is not renderable")
                }
            };

            let mut viewer = Viewer::new(env, td3, None, None);
            viewer.render();
        }
    }
}
