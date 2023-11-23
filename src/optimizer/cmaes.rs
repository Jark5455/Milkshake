use crate::device;
use crate::optimizer::MilkshakeOptimizer;
use clap::builder::TypedValueParser;
use std::ops::Deref;

// This is an implementation of barecmaes2.py from http://www.cmap.polytechnique.fr/~nikolaus.hansen/barecmaes2.py

pub struct CMAES {
    pub xmean: tch::nn::VarStore,
    pub sigma: f64,
    pub max_eval: u32,
    pub ftarget: f64,
    pub lambda: u32,
    pub mu: u32,
    pub weights: tch::Tensor,
}

impl CMAES {
    pub fn new(
        vs: std::rc::Rc<tch::nn::VarStore>,
        sigma: f64,
        max_eval: Option<u32>,
        ftarget: Option<f64>,
        popsize: Option<u32>,
    ) -> Self {
        let mut xmean = tch::nn::VarStore::new(**device);
        xmean
            .copy(vs.deref())
            .expect("CMAES failed to copy xstart varstore");

        let N = xmean.len() as u32;

        let max_eval = max_eval.unwrap_or(1e3 as u32 * N.pow(2));
        let ftarget = ftarget.unwrap_or(0f64);
        let popsize = popsize.unwrap_or(4 + (3f64 * (N as f64).ln()).floor() as u32);
        let lambda = popsize;
        let mu = lambda / 2;

        let mut weights_slice: Vec<f64> = (0..mu)
            .map(|i| (mu as f64 + 0.5f64).ln() - (i as f64 + 1f64).ln())
            .collect();
        let sum: f64 = weights_slice.iter().sum();
        weights_slice = weights_slice.iter().map(|w| w / sum).collect();

        let weights = tch::Tensor::from_slice(weights_slice.as_slice());

        Self {
            xmean,
            sigma,
            max_eval,
            ftarget,
            lambda,
            mu,
            weights,
        }
    }
}

impl MilkshakeOptimizer for CMAES {
    fn ask(&mut self) -> Vec<std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>> {
        todo!()
    }

    fn tell(
        &mut self,
        solutions: Vec<std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>>,
        losses: Vec<tch::Tensor>,
    ) {
        todo!()
    }

    fn result(&mut self) -> std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>> {
        todo!()
    }
}
