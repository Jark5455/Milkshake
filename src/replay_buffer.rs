use crate::device;

#[derive(Clone)]
pub struct ReplayBuffer {
    pub max_size: usize,
    pub ptr: usize,
    pub size: usize,
    pub state: Vec<Vec<f64>>,
    pub action: Vec<Vec<f64>>,
    pub next_state: Vec<Vec<f64>>,
    pub reward: Vec<f64>,
    pub not_done: Vec<f64>,
}

impl ReplayBuffer {
    pub fn new(state_dim: i64, action_dim: i64, max_size: Option<i64>) -> Self {
        let max_size = max_size.unwrap_or(1e6 as i64) as usize;

        ReplayBuffer {
            max_size,
            state: vec![vec![0f64; state_dim as usize]; max_size],
            action: vec![vec![0f64; action_dim as usize]; max_size],
            next_state: vec![vec![0f64; state_dim as usize]; max_size],
            reward: vec![0f64; max_size],
            not_done: vec![0f64; max_size],
            ptr: 0,
            size: 0,
        }
    }

    pub fn add(
        &mut self,
        state: Vec<f64>,
        action: Vec<f64>,
        next_state: Vec<f64>,
        reward: f64,
        done: f64,
    ) {
        self.state[self.ptr] = state;
        self.action[self.ptr] = action;
        self.next_state[self.ptr] = next_state;
        self.reward[self.ptr] = reward;
        self.not_done[self.ptr] = 1f64 - done;

        self.ptr = (self.ptr + 1) % self.max_size;
        self.size = std::cmp::min(self.size + 1, self.max_size);
    }

    pub fn sample(&self, batch_size: i64) -> Vec<tch::Tensor> {
        let mut rng = <rand::prelude::StdRng as rand::prelude::SeedableRng>::from_entropy();

        let mut sample_state = Vec::with_capacity(batch_size as usize);
        let mut sample_action = Vec::with_capacity(batch_size as usize);
        let mut sample_next_state = Vec::with_capacity(batch_size as usize);
        let mut sample_reward = Vec::with_capacity(batch_size as usize);
        let mut sample_not_done = Vec::with_capacity(batch_size as usize);

        for _ in 0..batch_size {
            let id = rand::prelude::Rng::gen_range(&mut rng, 0..self.size);

            sample_state.push(self.state[id].as_slice());
            sample_action.push(self.action[id].as_slice());
            sample_next_state.push(self.next_state[id].as_slice());
            sample_reward.push(self.reward[id]);
            sample_not_done.push(self.not_done[id]);
        }

        let sample_state_tensor = tch::Tensor::from_slice2(sample_state.as_slice())
            .totype(tch::Kind::Float)
            .to_device(**device);
        let sample_action_tensor = tch::Tensor::from_slice2(sample_action.as_slice())
            .totype(tch::Kind::Float)
            .to_device(**device);
        let sample_next_state_tensor = tch::Tensor::from_slice2(sample_next_state.as_slice())
            .totype(tch::Kind::Float)
            .to_device(**device);
        let sample_reward_tensor = tch::Tensor::from_slice(sample_reward.as_slice())
            .totype(tch::Kind::Float)
            .to_device(**device);
        let sample_not_done_tensor = tch::Tensor::from_slice(sample_not_done.as_slice())
            .totype(tch::Kind::Float)
            .to_device(**device);

        vec![
            sample_state_tensor,
            sample_action_tensor,
            sample_next_state_tensor,
            sample_reward_tensor,
            sample_not_done_tensor,
        ]
    }
}
