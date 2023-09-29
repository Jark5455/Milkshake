use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Borrow;
use std::fmt::Debug;
use std::ops::{Add, Index};
use tch::nn::{Module, Optimizer, OptimizerConfig};
use tch::Tensor;
use tch::{nn, Device, Reduction};

use crate::replay_buffer::ReplayBuffer;
use crate::{device, vs};

#[derive(Debug)]
struct Actor {
    pub layers: Vec<nn::Linear>,
    pub max_action: f64,
}

#[derive(Debug)]
struct Critic {
    pub q1_layers: Vec<nn::Linear>,
    pub q2_layers: Vec<nn::Linear>,
}

impl Actor {
    pub fn new(state_dim: i64, action_dim: i64, nn_shape: Vec<i64>, max_action: f64) -> Self {
        let mut shape = nn_shape.clone();
        shape.insert(0, state_dim);
        shape.insert(shape.len(), action_dim);

        let mut layers = Vec::new();

        for x in 1..nn_shape.len() {
            layers.push(nn::linear(
                vs.root(),
                nn_shape[x - 1],
                nn_shape[x],
                Default::default(),
            ));
        }

        Actor { layers, max_action }
    }
}

impl Module for Actor {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut alpha = self.layers[0].forward(xs).relu();

        for layer in &self.layers[..1] {
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
        map_serializer.serialize_entry("max_action", &self.max_action)?;

        for idx in 0..self.layers.len() {
            let layer_name = format!("layer_{}", idx);
            let cpu_t = self.layers[idx].ws.to_device(Device::Cpu);

            // tensor size
            let shape = cpu_t.size().clone();
            map_serializer.serialize_entry(
                format!("{}_tensor_shape", layer_name).as_str(),
                shape.as_slice(),
            )?;

            // tensor data
            let mut data: Vec<f64> =
                vec![0f64; shape.clone().iter().fold(1, |sum, val| sum * *val as usize)];
            self.layers[idx].ws.copy_data(
                data.as_mut_slice(),
                shape.clone().iter().fold(1, |sum, val| sum * *val as usize),
            );
            map_serializer.serialize_entry(
                format!("{}_tensor_data", layer_name).as_str(),
                data.as_slice(),
            )?;
        }

        map_serializer.end()
    }
}

impl<'de> Deserialize<'de> for Actor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        todo!()
    }
}

impl Critic {
    pub fn new(state_dim: i64, action_dim: i64, q1_shape: Vec<i64>, q2_shape: Vec<i64>) -> Self {
        let mut q1_shape = q1_shape.clone();
        q1_shape.insert(0, state_dim);
        q1_shape.insert(q1_shape.len(), action_dim);

        let mut q1_layers = Vec::new();

        for x in 1..q1_shape.len() {
            q1_layers.push(nn::linear(
                vs.root(),
                q1_shape[x - 1],
                q1_shape[x],
                Default::default(),
            ));
        }

        let mut q2_shape = q2_shape.clone();
        q2_shape.insert(0, state_dim);
        q2_shape.insert(q2_shape.len(), action_dim);

        let mut q2_layers = Vec::new();

        for x in 1..q2_shape.len() {
            q2_layers.push(nn::linear(
                vs.root(),
                q2_shape[x - 1],
                q2_shape[x],
                Default::default(),
            ));
        }

        Critic {
            q1_layers,
            q2_layers,
        }
    }
    pub fn Q1(&self, xs: &Tensor) -> Tensor {
        let mut alpha = self.q1_layers[0].forward(xs).relu();

        for layer in &self.q1_layers[..1] {
            alpha = layer.forward(&alpha).relu();
        }

        self.q1_layers.last().unwrap().forward(&alpha)
    }
}

impl Module for Critic {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let q1: Tensor;
        let q2: Tensor;

        {
            let mut alpha = self.q1_layers[0].forward(xs).relu();

            for layer in &self.q1_layers[..1] {
                alpha = layer.forward(&alpha).relu();
            }

            q1 = self.q1_layers.last().unwrap().forward(&alpha)
        }

        {
            let mut alpha = self.q2_layers[0].forward(xs).relu();

            for layer in &self.q2_layers[..1] {
                alpha = layer.forward(&alpha).relu();
            }

            q2 = self.q2_layers.last().unwrap().forward(&alpha)
        }

        Tensor::cat(&[q1, q2], 1)
    }
}

impl Serialize for Critic {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map_serializer = serializer.serialize_map(None)?;

        for idx in 0..self.q1_layers.len() {
            let layer_name = format!("q1_layer_{}", idx);
            let cpu_t = self.q1_layers[idx].ws.to_device(Device::Cpu);

            // tensor size
            let shape = cpu_t.size().clone();
            map_serializer.serialize_entry(
                format!("{}_tensor_shape", layer_name).as_str(),
                shape.as_slice(),
            )?;

            // tensor data
            let mut data: Vec<f64> =
                vec![0f64; shape.clone().iter().fold(1, |sum, val| sum * *val as usize)];
            self.q1_layers[idx].ws.copy_data(
                data.as_mut_slice(),
                shape.clone().iter().fold(1, |sum, val| sum * *val as usize),
            );
            map_serializer.serialize_entry(
                format!("{}_tensor_data", layer_name).as_str(),
                data.as_slice(),
            )?;
        }

        for idx in 0..self.q2_layers.len() {
            let layer_name = format!("q2_layer_{}", idx);
            let cpu_t = self.q2_layers[idx].ws.to_device(Device::Cpu);

            // tensor size
            let shape = cpu_t.size().clone();
            map_serializer.serialize_entry(
                format!("{}_tensor_shape", layer_name).as_str(),
                shape.as_slice(),
            )?;

            // tensor data
            let mut data: Vec<f64> =
                vec![0f64; shape.clone().iter().fold(1, |sum, val| sum * *val as usize)];
            self.q2_layers[idx].ws.copy_data(
                data.as_mut_slice(),
                shape.clone().iter().fold(1, |sum, val| sum * *val as usize),
            );
            map_serializer.serialize_entry(
                format!("{}_tensor_data", layer_name).as_str(),
                data.as_slice(),
            )?;
        }

        map_serializer.end()
    }
}

struct TD3 {
    pub actor: Actor,
    pub actor_target: Actor,
    pub actor_optimizer: Optimizer,
    pub critic: Critic,
    pub critic_target: Critic,
    pub critic_optimizer: Optimizer,
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
        let actor_optimizer = nn::Adam::default()
            .build(vs.borrow(), 3e-4)
            .expect("Failed to create Actor Optimizer");

        let critic = Critic::new(state_dim, action_dim, q1_shape.clone(), q2_shape.clone());
        let critic_target = Critic::new(state_dim, action_dim, q1_shape.clone(), q2_shape.clone());
        let critic_optimizer = nn::Adam::default()
            .build(vs.borrow(), 3e-4)
            .expect("Failed to create Critic Optimizer");

        TD3 {
            actor,
            actor_target,
            actor_optimizer,
            critic,
            critic_target,
            critic_optimizer,
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

    pub fn select_action(&self, action: Vec<f64>) -> Vec<f64> {
        let state = Tensor::from_slice(action.as_slice()).to_device(**device);
        let tensor = self.actor.forward(&state).to_device(Device::Cpu);
        let len = tensor
            .size()
            .clone()
            .iter()
            .fold(1, |sum, val| sum * *val as usize);

        let mut vec = vec![0f64; len];
        tensor.copy_data(vec.as_mut_slice(), len);

        vec
    }

    pub fn train(&mut self, replay_buffer: ReplayBuffer, batch_size: Option<i64>) {
        let batch_size = batch_size.unwrap_or(256);
        let samples = replay_buffer.sample(batch_size);

        let target_q = tch::no_grad(|| {
            let noise = samples[1]
                .rand_like()
                .multiply_scalar(self.policy_noise)
                .clamp(-self.noise_clip, self.noise_clip);
            let next_action = self
                .actor_target
                .forward(&samples[2])
                .add(noise)
                .clamp(-self.max_action, self.max_action);

            let q = self
                .critic_target
                .forward(&Tensor::cat(&[&samples[2], &next_action], 1));
            let split_q = q.split(batch_size, 1);

            let target_q1 = &split_q[0];
            let target_q2 = &split_q[1];

            let min_q = target_q1.min_other(target_q2);

            samples
                .index(3)
                .add(samples[4].multiply(&min_q).multiply_scalar(self.discount))
        });

        let q = self
            .critic
            .forward(&Tensor::cat(&[&samples[0], &samples[1]], 1));
        let split_q = q.split(batch_size, 1);

        let current_q1 = &split_q[0];
        let current_q2 = &split_q[1];

        let critic_loss = current_q1
            .mse_loss(&target_q, Reduction::None)
            .add(current_q2.mse_loss(&target_q, Reduction::None));

        self.critic_optimizer.zero_grad();
        critic_loss.backward();
        self.critic_optimizer.step();

        if self.total_it % self.policy_freq == 0 {
            let actor_loss = -self.critic.Q1(&Tensor::cat(
                &[&samples[0], &self.actor.forward(&samples[0])],
                1,
            ));

            self.actor_optimizer.zero_grad();
            actor_loss.backward();
            self.actor_optimizer.step();

            for (param, target_param) in self
                .critic
                .q1_layers
                .iter_mut()
                .zip(self.critic_target.q1_layers.iter_mut())
            {
                param.ws.copy_(&target_param.ws);
            }

            for (param, target_param) in self
                .critic
                .q2_layers
                .iter_mut()
                .zip(self.critic_target.q2_layers.iter_mut())
            {
                param.ws.copy_(&target_param.ws);
            }

            for (param, target_param) in self
                .actor
                .layers
                .iter_mut()
                .zip(self.actor_target.layers.iter_mut())
            {
                param.ws.copy_(&target_param.ws);
            }
        }
    }

    pub fn save() {}
}
