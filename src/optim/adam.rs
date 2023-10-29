extern crate tch;

use crate::device;
use crate::td3::TD3;

pub struct AdamOpt {
    pub td3: std::rc::Rc<TD3>,
    pub opt: tch::nn::Optimizer,
    pub vs: tch::nn::VarStore
}

impl AdamOpt {
    pub fn new(td3: std::rc::Rc<TD3>, lr: Option<f64>) -> AdamOpt {
        let lr = lr.unwrap_or(3e-4);
        let vs = tch::nn::VarStore::new(**device);

        let opt = tch::nn::OptimizerConfig::build(tch::nn::Adam::default(), &vs, lr)
            .expect("Failed to create ADAM Optimizer");

        Self {
            td3,
            opt,
            vs
        }
    }
}