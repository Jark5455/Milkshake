use crate::device;
use crate::optimizer::MilkshakeOptimizer;
use std::ops::Mul;

// This is an implementation of barecmaes2.py from http://www.cmap.polytechnique.fr/~nikolaus.hansen/barecmaes2.py

struct BestSolution {
    pub x: Option<tch::Tensor>,
    pub f: Option<tch::Tensor>,
    pub evals: Option<tch::Tensor>,
}

impl BestSolution {
    pub fn new(x: Option<tch::Tensor>, f: Option<tch::Tensor>, evals: Option<tch::Tensor>) -> Self {
        Self { x, f, evals }
    }

    pub fn update(&mut self, arx: tch::Tensor, arf: tch::Tensor, evals: Option<tch::Tensor>) {
        if self.f == None
            || arf
                .min()
                .le(self.f.as_ref().unwrap().f_double_value(&[0]).unwrap())
                .f_int64_value(&[0])
                .unwrap()
                == 1
        {
            let i = arf.index(&[Some(arf.min())]).f_int64_value(&[0]).unwrap();
            self.x = Some(arx.get(i));
            self.f = Some(arf.get(i));

            if self.evals != None {
                self.evals = Some(evals.unwrap() - arf.size()[0] + i + 1);
            }
        }
    }
}

pub struct CMAES {
    pub xmean: tch::nn::VarStore,
    pub sigma: f64,
    pub max_eval: u32,
    pub ftarget: f64,
    pub lambda: u32,
    pub mu: u32,
    pub weights: tch::Tensor,
    pub mueff: f64,
    pub cc: f64,
    pub cs: f64,
    pub c1: f64,
    pub cmu: f64,
    pub damps: f64,
    pub pc: tch::Tensor,
    pub ps: tch::Tensor,
    pub B: tch::Tensor,
    pub D: tch::Tensor,
    pub C: tch::Tensor,
    pub invSqrtC: tch::Tensor,
    pub eigeneval: u32,
    pub counteval: u32,
    pub fitvals: tch::Tensor,

    bestsolution: BestSolution,
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
            .copy(vs.as_ref())
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
        let mueff: f64 = weights_slice.iter().sum::<f64>().powi(2)
            / (weights_slice.iter().map(|w| w.powi(2)).sum::<f64>());

        let weights = tch::Tensor::from_slice(weights_slice.as_slice());

        let cc = (4f64 + mueff / N as f64) / (N as f64 + 4f64 + 2f64 * mueff / N as f64);
        let cs = (mueff + 2f64) / (N as f64 + mueff + 5f64);
        let c1 = 2f64 / ((N as f64 + 1.3f64).powi(2) + mueff);
        let cmu = f64::min(
            1f64 - c1,
            2f64 * (mueff - 2f64 + 1f64 / mueff) / ((N as f64 + 2f64).powi(2) + mueff),
        );
        let damps = 2f64 * (mu as f64 / lambda as f64) + 0.3f64 + cs;

        let pc = tch::Tensor::from_slice(vec![0; N as usize].as_slice()).to_device(**device);
        let ps = tch::Tensor::from_slice(vec![0; N as usize].as_slice()).to_device(**device);

        let B = tch::Tensor::eye(N as i64, (tch::Kind::Float, **device));
        let D = tch::Tensor::from_slice(vec![1; N as usize].as_slice()).to_device(**device);
        let C = tch::Tensor::eye(N as i64, (tch::Kind::Float, **device));
        let invSqrtC = tch::Tensor::eye(N as i64, (tch::Kind::Float, **device));

        let eigeneval = 0;
        let counteval = 0;
        let fitvals = tch::Tensor::new();

        let bestsolution = BestSolution::new(None, None, None);

        Self {
            xmean,
            sigma,
            max_eval,
            ftarget,
            lambda,
            mu,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            pc,
            ps,
            B,
            D,
            C,
            invSqrtC,
            eigeneval,
            counteval,
            fitvals,
            bestsolution,
        }
    }
}

impl MilkshakeOptimizer for CMAES {
    fn ask(&mut self) -> Vec<std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>> {
        if (self.counteval - self.eigeneval) as f64
            > self.lambda as f64 / (self.c1 + self.cmu) / self.C.size1().unwrap() as f64 / 10f64
        {
            self.eigeneval = self.counteval;
            (self.B, self.D) = self.C.linalg_eigh("L");
            self.D = self.D.sqrt();
            self.invSqrtC = self.B.copy()
                * self.D.pow(&tch::Tensor::from_slice(&[-1])).diag(0)
                * self.B.transpose(0, 1);
        }

        let mut res = vec![];

        for k in 0..self.lambda {
            let mut x = tch::nn::VarStore::new(**device);
            x.copy(&self.xmean).unwrap();

            let z = self.D.copy().normal_(0f64, 1f64).mul(&self.D.copy());
        }

        res
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
