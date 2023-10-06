use anyhow::Result;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::any::Any;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{Cursor, Read, Write};
use std::ops::{Add, Index};
use std::{mem, slice};
use tch::nn::{Module, OptimizerConfig};
use tch::{nn, Device, Reduction};
use tch::{Kind, Tensor};

use crate::device;
use crate::replay_buffer::ReplayBuffer;

pub struct WrappedLayer {
    pub layer: nn::Linear,
    pub input: i64,
    pub output: i64,
}

impl WrappedLayer {
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        self.layer.forward(xs)
    }
}

pub struct Actor {
    pub vs: nn::VarStore,
    pub opt: nn::Optimizer,
    pub layers: Vec<WrappedLayer>,

    pub max_action: f64,
}

pub struct Critic {
    pub vs: nn::VarStore,
    pub opt: nn::Optimizer,
    pub q1_layers: Vec<WrappedLayer>,
    pub q2_layers: Vec<WrappedLayer>,
}

impl Actor {
    pub fn new(state_dim: i64, action_dim: i64, nn_shape: Vec<i64>, max_action: f64) -> Self {
        let vs = nn::VarStore::new(**device);

        let mut shape = nn_shape.clone();
        shape.insert(0, state_dim);
        shape.insert(shape.len(), action_dim);

        let mut layers = Vec::new();

        for x in 1..shape.len() {
            layers.push(WrappedLayer {
                layer: nn::linear(vs.root(), shape[x - 1], shape[x], Default::default()),

                input: shape[x - 1],
                output: shape[x],
            });
        }

        let opt = nn::Adam::default()
            .build(&vs, 3e-4)
            .expect("Failed to create Actor Optimizer");

        Actor {
            vs,
            opt,
            layers,
            max_action,
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let f_xs = xs.totype(Kind::Float);
        let mut alpha = self.layers[0].forward(&f_xs).relu();

        for layer in &self.layers[1..1] {
            alpha = layer.forward(&alpha).relu();
        }

        self.layers
            .last()
            .unwrap()
            .forward(&alpha)
            .tanh()
            .multiply_scalar(self.max_action)
    }
}

impl Serialize for Actor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map_serializer = serializer.serialize_map(None)?;
        map_serializer.serialize_entry("max_action", self.max_action.to_string().as_bytes())?;
        map_serializer.serialize_entry("num_layers", self.layers.len().to_string().as_bytes())?;

        for idx in 0..self.layers.len() {
            map_serializer.serialize_entry(
                format!("actor_layer_{}_input_dim", idx).as_str(),
                &self.layers[idx].input.to_string().as_bytes(),
            )?;
            map_serializer.serialize_entry(
                format!("actor_layer_{}_output_dim", idx).as_str(),
                &self.layers[idx].output.to_string().as_bytes(),
            )?;
        }

        let mut cursor = Cursor::new(Vec::<u8>::new());

        self.vs
            .save_to_stream(&mut cursor)
            .expect("Failed to save varstore to byte buffer");

        map_serializer.serialize_entry("actor_varstore", cursor.into_inner().as_slice())?;
        map_serializer.end()
    }
}

impl<'de> Deserialize<'de> for Actor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map: HashMap<String, Vec<u8>> = Deserialize::deserialize(deserializer)?;

        let max_action_string = String::from_utf8(
            map.get("max_action")
                .expect("Failed to get key max_action")
                .clone(),
        )
        .expect("max_action not valid utf8 string");
        let max_action = max_action_string
            .parse()
            .expect("Failed to parse max_action");

        let num_layers_string = String::from_utf8(
            map.get("num_layers")
                .expect("Failed to get key num_layers")
                .clone(),
        )
        .expect("num_layers not valid utf8 string");
        let num_layers: i64 = num_layers_string
            .parse()
            .expect("Failed to parse num_layers");

        let mut vs = nn::VarStore::new(**device);

        let mut layers = Vec::new();
        for x in 0..num_layers {
            let input_dim_string = String::from_utf8(
                map.get(format!("actor_layer_{}_input_dim", x).as_str())
                    .expect(format!("Failed to get key actor_layer_{}_input_dim", x).as_str())
                    .clone(),
            )
            .expect(format!("actor_layer_{}_input_dim is not a valid utf8 string", x).as_str());
            let input_dim: i64 = input_dim_string
                .parse()
                .expect(format!("Failed to parse actor_layer_{}_input_dim", x).as_str());

            let output_dim_string = String::from_utf8(
                map.get(format!("actor_layer_{}_output_dim", x).as_str())
                    .expect(format!("Failed to get key actor_layer_{}_output_dim", x).as_str())
                    .clone(),
            )
            .expect(format!("actor_layer_{}_output_dim is not a valid utf8 string", x).as_str());
            let output_dim: i64 = output_dim_string
                .parse()
                .expect(format!("Failed to parse actor_layer_{}_output_dim", x).as_str());

            layers.push(WrappedLayer {
                layer: nn::linear(vs.root(), input_dim, output_dim, Default::default()),

                input: input_dim,
                output: output_dim,
            });
        }

        let opt = nn::Adam::default()
            .build(&vs, 3e-4)
            .expect("Failed to create Actor Optimizer");

        let varstorestring = map
            .get("actor_varstore")
            .expect("actor_varstore not found")
            .clone();

        vs.load_from_stream(Cursor::new(varstorestring))
            .expect("Failed to load varstore from string");

        Ok(Actor {
            vs,
            opt,
            layers,
            max_action,
        })
    }
}

impl Critic {
    pub fn new(state_dim: i64, action_dim: i64, q1_shape: Vec<i64>, q2_shape: Vec<i64>) -> Self {
        let vs = nn::VarStore::new(**device);

        let mut q1_shape = q1_shape.clone();
        q1_shape.insert(0, state_dim + action_dim);
        q1_shape.insert(q1_shape.len(), 1);

        let mut q1_layers = Vec::new();

        for x in 1..q1_shape.len() {
            q1_layers.push(WrappedLayer {
                layer: nn::linear(vs.root(), q1_shape[x - 1], q1_shape[x], Default::default()),

                input: q1_shape[x - 1],
                output: q1_shape[x],
            });
        }

        let mut q2_shape = q2_shape.clone();
        q2_shape.insert(0, state_dim + action_dim);
        q2_shape.insert(q2_shape.len(), 1);

        let mut q2_layers = Vec::new();

        for x in 1..q2_shape.len() {
            q2_layers.push(WrappedLayer {
                layer: nn::linear(vs.root(), q2_shape[x - 1], q2_shape[x], Default::default()),

                input: q2_shape[x - 1],
                output: q2_shape[x],
            });
        }

        let opt = nn::Adam::default()
            .build(&vs, 3e-4)
            .expect("Failed to create Critic Optimizer");

        Critic {
            vs,
            opt,
            q1_layers,
            q2_layers,
        }
    }
    pub fn Q1(&self, xs: &Tensor) -> Tensor {
        let mut alpha = self.q1_layers[0].forward(xs).relu();

        for layer in &self.q1_layers[1..1] {
            alpha = layer.forward(&alpha).relu();
        }

        self.q1_layers.last().unwrap().forward(&alpha)
    }

    fn forward(&self, state: &Tensor, action: &Tensor) -> (Tensor, Tensor) {
        let xs = Tensor::cat(&[state, action], 1);

        let q1: Tensor;
        let q2: Tensor;

        {
            let mut alpha = self.q1_layers[0].forward(&xs).relu();

            for layer in &self.q1_layers[1..1] {
                alpha = layer.forward(&alpha).relu();
            }

            q1 = self.q1_layers.last().unwrap().forward(&alpha)
        }

        {
            let mut alpha = self.q2_layers[0].forward(&xs).relu();

            for layer in &self.q2_layers[1..1] {
                alpha = layer.forward(&alpha).relu();
            }

            q2 = self.q2_layers.last().unwrap().forward(&alpha)
        }

        (q1, q2)
    }
}

impl Serialize for Critic {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map_serializer = serializer.serialize_map(None)?;
        map_serializer
            .serialize_entry("num_q1_layers", self.q1_layers.len().to_string().as_bytes())?;
        map_serializer
            .serialize_entry("num_q2_layers", self.q2_layers.len().to_string().as_bytes())?;

        for idx in 0..self.q1_layers.len() {
            map_serializer.serialize_entry(
                format!("q1_layer_{}_input_dim", idx).as_str(),
                self.q1_layers[idx].input.to_string().as_bytes(),
            )?;
            map_serializer.serialize_entry(
                format!("q1_layer_{}_output_dim", idx).as_str(),
                self.q1_layers[idx].output.to_string().as_bytes(),
            )?;
        }

        for idx in 0..self.q2_layers.len() {
            map_serializer.serialize_entry(
                format!("q2_layer_{}_input_dim", idx).as_str(),
                self.q2_layers[idx].input.to_string().as_bytes(),
            )?;
            map_serializer.serialize_entry(
                format!("q2_layer_{}_output_dim", idx).as_str(),
                self.q2_layers[idx].output.to_string().as_bytes(),
            )?;
        }

        let mut cursor = Cursor::new(Vec::<u8>::new());
        self.vs
            .save_to_stream(&mut cursor)
            .expect("Failed to save varstore to cursor");

        map_serializer.serialize_entry("critic_varstore", cursor.into_inner().as_slice())?;
        map_serializer.end()
    }
}

impl<'de> Deserialize<'de> for Critic {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map: HashMap<String, Vec<u8>> = Deserialize::deserialize(deserializer)?;

        let num_q1_layers_string = String::from_utf8(
            map.get("num_q1_layers")
                .expect("Failed to get key num_q1_layers")
                .clone(),
        )
        .expect("num_q1_layers not valid utf8 string");
        let num_q1_layers: i64 = num_q1_layers_string
            .parse()
            .expect("Failed to parse num_layers");

        let num_q2_layers_string = String::from_utf8(
            map.get("num_q2_layers")
                .expect("Failed to get key num_q2_layers")
                .clone(),
        )
        .expect("num_q2_layers not valid utf8 string");
        let num_q2_layers: i64 = num_q2_layers_string
            .parse()
            .expect("Failed to parse num_layers");

        let mut vs = nn::VarStore::new(**device);

        let mut q1_layers = Vec::new();
        let mut q2_layers = Vec::new();

        for x in 0..num_q1_layers {
            let input_dim_string = String::from_utf8(
                map.get(format!("q1_layer_{}_input_dim", x).as_str())
                    .expect(format!("Failed to get key q1_layer_{}_input_dim", x).as_str())
                    .clone(),
            )
            .expect(format!("q1_layer_{}_input_dim is not a valid utf8 string", x).as_str());
            let input_dim: i64 = input_dim_string
                .parse()
                .expect(format!("Failed to parse q1_layer_{}_input_dim", x).as_str());

            let output_dim_string = String::from_utf8(
                map.get(format!("q1_layer_{}_output_dim", x).as_str())
                    .expect(format!("Failed to get key q1_layer_{}_output_dim", x).as_str())
                    .clone(),
            )
            .expect(format!("q1_layer_{}_output_dim is not a valid utf8 string", x).as_str());
            let output_dim: i64 = output_dim_string
                .parse()
                .expect(format!("Failed to parse q1_layer_{}_output_dim", x).as_str());

            q1_layers.push(WrappedLayer {
                layer: nn::linear(vs.root(), input_dim, output_dim, Default::default()),

                input: input_dim,
                output: output_dim,
            });
        }

        for x in 0..num_q2_layers {
            let input_dim_string = String::from_utf8(
                map.get(format!("q2_layer_{}_input_dim", x).as_str())
                    .expect(format!("Failed to get key q2_layer_{}_input_dim", x).as_str())
                    .clone(),
            )
            .expect(format!("q2_layer_{}_input_dim is not a valid utf8 string", x).as_str());
            let input_dim: i64 = input_dim_string
                .parse()
                .expect(format!("Failed to parse q2_layer_{}_input_dim", x).as_str());

            let output_dim_string = String::from_utf8(
                map.get(format!("q2_layer_{}_output_dim", x).as_str())
                    .expect(format!("Failed to get key q2_layer_{}_output_dim", x).as_str())
                    .clone(),
            )
            .expect(format!("q2_layer_{}_output_dim is not a valid utf8 string", x).as_str());
            let output_dim: i64 = output_dim_string
                .parse()
                .expect(format!("Failed to parse q2_layer_{}_output_dim", x).as_str());

            q2_layers.push(WrappedLayer {
                layer: nn::linear(vs.root(), input_dim, output_dim, Default::default()),

                input: input_dim,
                output: output_dim,
            });
        }

        let opt = nn::Adam::default()
            .build(&vs, 3e-4)
            .expect("Failed to create Critic Optimizer");

        let varstorestring = map
            .get("critic_varstore")
            .expect("critic_varstore not found")
            .clone();

        vs.load_from_stream(Cursor::new(varstorestring))
            .expect("Failed to load varstore from string");

        Ok(Critic {
            vs,
            opt,
            q1_layers,
            q2_layers,
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct TD3 {
    pub actor: Actor,
    pub actor_target: Actor,
    pub critic: Critic,
    pub critic_target: Critic,
    pub action_dim: i64,
    pub state_dim: i64,
    pub max_action: f64,
    pub tau: f64,
    pub discount: f64,
    pub policy_noise: f64,
    pub noise_clip: f64,
    pub policy_freq: i64,
    pub total_it: i64,
}

impl TD3 {
    pub fn new(
        state_dim: i64,
        action_dim: i64,
        max_action: f64,
        actor_shape: Option<Vec<i64>>,
        q1_shape: Option<Vec<i64>>,
        q2_shape: Option<Vec<i64>>,
        tau: Option<f64>,
        discount: Option<f64>,
        policy_noise: Option<f64>,
        noise_clip: Option<f64>,
        policy_freq: Option<i64>,
    ) -> Self {
        let actor_shape = actor_shape.unwrap_or(vec![256, 256, 256]);
        let q1_shape = q1_shape.unwrap_or(vec![256, 256, 256]);
        let q2_shape = q2_shape.unwrap_or(vec![256, 256, 256]);

        let tau = tau.unwrap_or(0.005);
        let discount = discount.unwrap_or(0.99);
        let policy_noise = policy_noise.unwrap_or(0.2);
        let noise_clip = noise_clip.unwrap_or(0.5);
        let policy_freq = policy_freq.unwrap_or(2);

        let actor = Actor::new(state_dim, action_dim, actor_shape.clone(), max_action);
        let actor_target = Actor::new(state_dim, action_dim, actor_shape.clone(), max_action);

        let critic = Critic::new(state_dim, action_dim, q1_shape.clone(), q2_shape.clone());
        let critic_target = Critic::new(state_dim, action_dim, q1_shape.clone(), q2_shape.clone());

        TD3 {
            actor,
            actor_target,
            critic,
            critic_target,
            action_dim,
            state_dim,
            max_action,
            tau,
            discount,
            policy_noise,
            noise_clip,
            policy_freq,
            total_it: 0,
        }
    }

    pub fn select_action(&self, state: Vec<f64>) -> Vec<f64> {
        let state = Tensor::from_slice(&state).to_device(**device);
        let tensor = self.actor.forward(&state).to_device(Device::Cpu);
        let len = tensor.size().iter().fold(1, |sum, val| sum * *val as usize);

        let mut vec = vec![0f32; len];
        tensor.copy_data(vec.as_mut_slice(), len);

        vec.iter().map(|x| *x as f64).collect()
    }

    pub fn train(&mut self, replay_buffer: &ReplayBuffer, batch_size: Option<i64>) {
        let batch_size = batch_size.unwrap_or(256);
        let samples = replay_buffer.sample(batch_size);

        let state = &samples[0];
        let action = &samples[1];
        let next_state = &samples[2];
        let reward = &samples[3];
        let done = &samples[4];

        let target_q = tch::no_grad(|| {
            let noise = action
                .rand_like()
                .multiply_scalar(self.policy_noise)
                .clamp(-self.noise_clip, self.noise_clip);

            let next_action = self
                .actor_target
                .forward(next_state)
                .add(noise)
                .clamp(-self.max_action, self.max_action);

            let q = self.critic_target.forward(next_state, &next_action);

            let target_q1 = &q.0;
            let target_q2 = &q.1;

            let min_q = target_q1.min_other(target_q2);

            reward.unsqueeze(1) + ((done.unsqueeze(1) * min_q) * self.discount)
        });

        let q = self.critic.forward(state, action);

        let current_q1 = &q.0;
        let current_q2 = &q.1;

        let q1_loss = current_q1.mse_loss(&target_q, Reduction::None);
        let q2_loss = current_q2.mse_loss(&target_q, Reduction::None);

        let critic_loss = (q1_loss + q2_loss).sum(Kind::Float);

        self.critic.opt.zero_grad();
        critic_loss.backward();
        self.critic.opt.step();

        if self.total_it % self.policy_freq == 0 {
            let actor_loss = -self
                .critic
                .Q1(&Tensor::cat(&[state, &self.actor.forward(state)], 1))
                .mean(Kind::Float);

            self.actor.opt.zero_grad();
            actor_loss.backward();
            self.actor.opt.step();

            tch::no_grad(|| {
                for (param, target_param) in self
                    .actor
                    .vs
                    .trainable_variables()
                    .iter_mut()
                    .zip(self.actor_target.vs.trainable_variables().iter_mut())
                {
                    target_param.copy_(
                        &(param.multiply_scalar(self.tau)
                            + (target_param.copy().multiply_scalar(1f64 - self.tau))),
                    );
                }

                for (param, target_param) in self
                    .critic
                    .vs
                    .trainable_variables()
                    .iter_mut()
                    .zip(self.critic_target.vs.trainable_variables().iter_mut())
                {
                    target_param.copy_(
                        &(param.multiply_scalar(self.tau)
                            + (target_param.copy().multiply_scalar(1f64 - self.tau))),
                    );
                }
            })
        }
    }
}
