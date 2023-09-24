use rand::distributions::uniform::SampleBorrow;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;
use tch::nn;
use tch::nn::Module;
use tch::Tensor;

use crate::vs;

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

struct TD3 {}

impl TD3 {
    pub fn new(
        state_dim: i64,
        action_dim: i64,
        max_action: f64,
        actor_shape: Option<Vec<i64>>,
        q1_shape: Option<Vec<i64>>,
        q2_shape: Option<Vec<i64>>,
        tau: Option<f64>,
        policy_noise: Option<f64>,
        noise_clip: Option<f64>,
        policy_freq: Option<i64>,
    ) {
        let actor_shape = actor_shape.unwrap_or(vec![256, 256, 256]);
        let q1_shape = q1_shape.unwrap_or(vec![256, 256, 256]);
        let q2_shape = q2_shape.unwrap_or(vec![256, 256, 256]);

        let actor = Actor::new(state_dim, action_dim, actor_shape, max_action);
        let critic = Critic::new(state_dim, action_dim, q1_shape, q2_shape);
    }
}
