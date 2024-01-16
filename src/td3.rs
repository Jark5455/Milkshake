use crate::device;
use crate::optimizer::adam::ADAM;
use crate::replay_buffer::ReplayBuffer;

use crate::optimizer::MilkshakeOptimizer;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use std::fmt::Debug;

use std::ops::{Add, Deref};
use tch::nn::Module;
use tch::{Device, Reduction};
use tch::{Kind, Tensor};

#[derive(Debug)]
pub struct MilkshakeLayer {
    pub layer: tch::nn::Linear,
    pub input: i64,
    pub output: i64,
}

impl Module for MilkshakeLayer {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.layer.forward(xs)
    }
}

#[derive(Debug)]
pub struct MilkshakeNetwork {
    pub layers: Vec<MilkshakeLayer>,
}

impl Module for MilkshakeNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut alpha = self.layers.first().unwrap().forward(&xs.totype(Kind::Float)).relu();

        for layer in &self.layers[1..1] {
            alpha = layer.forward(&alpha).relu();
        }

        self.layers.last().unwrap().forward(&alpha).tanh()
    }
}

pub struct Actor {
    pub vs: std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>,
    pub actor: MilkshakeNetwork,
    pub max_action: f64,
}

pub struct Critic {
    pub vs: std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>,
    pub q1: MilkshakeNetwork,
    pub q2: MilkshakeNetwork,
}

impl Actor {
    pub fn new(state_dim: i64, action_dim: i64, nn_shape: Vec<i64>, max_action: f64) -> Self {
        let vs = std::rc::Rc::new(std::cell::RefCell::new(tch::nn::VarStore::new(**device)));

        let mut shape = nn_shape.clone();
        shape.insert(0, state_dim);
        shape.insert(shape.len(), action_dim);

        let mut layers = Vec::new();

        for x in 1..shape.len() {
            layers.push(MilkshakeLayer {
                layer: tch::nn::linear(
                    vs.borrow().root(),
                    shape[x - 1],
                    shape[x],
                    Default::default(),
                ),

                input: shape[x - 1],
                output: shape[x],
            });
        }

        let actor = MilkshakeNetwork { layers };

        Actor {
            vs,
            actor,
            max_action,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        self.actor.forward(xs)
    }
}

impl Serialize for Actor {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        todo!()
    }
}

impl<'de> Deserialize<'de> for Actor {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        todo!()
    }
}

impl Critic {
    pub fn new(state_dim: i64, action_dim: i64, q1_shape: Vec<i64>, q2_shape: Vec<i64>) -> Self {
        let vs = std::rc::Rc::new(std::cell::RefCell::new(tch::nn::VarStore::new(**device)));

        let mut q1_shape = q1_shape.clone();
        q1_shape.insert(0, state_dim + action_dim);
        q1_shape.insert(q1_shape.len(), 1);

        let mut q1_layers = Vec::new();

        for x in 1..q1_shape.len() {
            q1_layers.push(MilkshakeLayer {
                layer: tch::nn::linear(
                    vs.borrow().root(),
                    q1_shape[x - 1],
                    q1_shape[x],
                    Default::default(),
                ),

                input: q1_shape[x - 1],
                output: q1_shape[x],
            });
        }

        let mut q2_shape = q2_shape.clone();
        q2_shape.insert(0, state_dim + action_dim);
        q2_shape.insert(q2_shape.len(), 1);

        let mut q2_layers = Vec::new();

        for x in 1..q2_shape.len() {
            q2_layers.push(MilkshakeLayer {
                layer: tch::nn::linear(
                    vs.borrow().root(),
                    q2_shape[x - 1],
                    q2_shape[x],
                    Default::default(),
                ),

                input: q2_shape[x - 1],
                output: q2_shape[x],
            });
        }

        let q1 = MilkshakeNetwork { layers: q1_layers };
        let q2 = MilkshakeNetwork { layers: q2_layers };

        Critic { vs, q1, q2 }
    }

    pub fn forward(&self, state: &Tensor, action: &Tensor) -> (Tensor, Tensor) {
        let xs = Tensor::cat(&[state, action], 1);

        let q1 = self.q1.forward(&xs);
        let q2 = self.q2.forward(&xs);

        (q1, q2)
    }

    pub fn Q1(&self, xs: &Tensor) -> Tensor {
        self.q1.forward(xs)
    }
}

impl Serialize for Critic {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        todo!()
    }
}

impl<'de> Deserialize<'de> for Critic {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        todo!()
    }
}
pub struct TD3 {
    actor: Actor,
    actor_target: Actor,
    critic: Critic,
    critic_target: Critic,

    actor_opt: Box<dyn MilkshakeOptimizer>,
    critic_opt: Box<dyn MilkshakeOptimizer>,

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

        let actor_opt = Box::new(ADAM::new(3e-4, actor.vs.clone()));
        let critic_opt = Box::new(ADAM::new(3e-4, critic.vs.clone()));

        TD3 {
            actor,
            actor_target,
            critic,
            critic_target,
            actor_opt,
            critic_opt,
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


        let grads = self.critic_opt.grads();
        let mut critic_train_closure = || {
            let solutions = self.critic_opt.ask();
            let mut losses = vec![];

            for solution in &solutions {

                if !std::rc::Rc::ptr_eq(solution, &self.critic.vs) {
                    self.critic
                        .vs
                        .borrow_mut()
                        .copy(solution.borrow().deref())
                        .expect("Failed to copy test solution to critic");
                }

                let q = self.critic.forward(state, action);

                let current_q1 = &q.0;
                let current_q2 = &q.1;

                let q1_loss = current_q1.mse_loss(&target_q, Reduction::Mean);
                let q2_loss = current_q2.mse_loss(&target_q, Reduction::Mean);

                let critic_loss = q1_loss + q2_loss;
                losses.push(critic_loss);
            }

            self.critic_opt.tell(solutions, losses);

            let critic_result = self.critic_opt.result();
            if !std::rc::Rc::ptr_eq(&critic_result, &self.critic.vs) {
                self.critic
                    .vs
                    .borrow_mut()
                    .copy(critic_result.borrow().deref())
                    .expect("Failed to copy result to critic from optimizer");
            }
        };

        match grads {
            true => critic_train_closure(),
            false => tch::no_grad(critic_train_closure),
        }

        if self.total_it % self.policy_freq == 0 {

            let grads = self.actor_opt.grads();
            let mut actor_train_closure = || {
                let solutions = self.actor_opt.ask();
                let mut losses = vec![];

                for solution in &solutions {

                    if !std::rc::Rc::ptr_eq(solution, &self.actor.vs) {
                        self.actor
                            .vs
                            .borrow_mut()
                            .copy(solution.borrow().deref())
                            .expect("Failed to copy test solution to actor");
                    }

                    let loss = -self
                        .critic
                        .Q1(&Tensor::cat(&[state, &self.actor.forward(state)], 1))
                        .mean(Kind::Float);

                    losses.push(loss);
                }

                self.actor_opt.tell(solutions, losses);

                let actor_result = self.actor_opt.result();
                if !std::rc::Rc::ptr_eq(&actor_result, &self.actor.vs) {
                    self.actor
                        .vs
                        .borrow_mut()
                        .copy(actor_result.borrow().deref())
                        .expect("Failed to copy result to actor from optimizer");
                }
            };

            match grads {
                true => actor_train_closure(),
                false => tch::no_grad(actor_train_closure),
            }

            tch::no_grad(|| {
                for (param, target_param) in self
                    .actor
                    .vs
                    .borrow_mut()
                    .trainable_variables()
                    .iter_mut()
                    .zip(
                        self.actor_target
                            .vs
                            .borrow_mut()
                            .trainable_variables()
                            .iter_mut(),
                    )
                {
                    target_param.copy_(
                        &(param.multiply_scalar(self.tau)
                            + (target_param.copy().multiply_scalar(1f64 - self.tau))),
                    );
                }

                for (param, target_param) in self
                    .critic
                    .vs
                    .borrow_mut()
                    .trainable_variables()
                    .iter_mut()
                    .zip(
                        self.critic_target
                            .vs
                            .borrow_mut()
                            .trainable_variables()
                            .iter_mut(),
                    )
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

impl Serialize for TD3 {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        todo!()
    }
}

impl<'de> Deserialize<'de> for TD3 {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        todo!()
    }
}
