use crate::device;
use crate::optimizer::MilkshakeOptimizer;
use crate::optimizer::RefVs;

use tch::IndexOp;

// This is a barebones CMAES implementation in pytorch

pub struct CMAES {
    pub vs: RefVs,

    pub cc: f64,
    pub cs: f64,
    pub c1: f64,
    pub cmu: f64,

    pub sigma: f64,
    pub xmean: tch::Tensor,
    pub variation: tch::Tensor,
    pub newpop: Vec<std::rc::Rc<std::cell::RefCell<tch::Tensor>>>,
    pub N: i64,
    pub lambda: i64,
    pub mu: i64,
    pub weights: tch::Tensor,
    pub mueff: f64,
    pub damps: f64,
    pub B: tch::Tensor,
    pub D: tch::Tensor,
    pub Dinv: tch::Tensor,
    pub C: tch::Tensor,
    pub Cold: tch::Tensor,
    pub invsqrtC: tch::Tensor,
    pub pc: tch::Tensor,
    pub ps: tch::Tensor,

    pub counteval: i64,
    pub eigeneval: i64,
    pub gen: i64,
}

impl CMAES {
    pub fn new(vs: RefVs, sigma: Option<f64>, popsize: Option<i64>) -> Self {
        let sigma = sigma.unwrap_or(0.5f64);

        let xmean = Self::vs_to_flattensor(vs.clone());

        let N = xmean.size()[0];

        let lambda = popsize.unwrap_or(4 + (3f64 * (N as f64).ln()).floor() as i64);
        let mu = lambda / 2;

        let mut weights = vec![0f64; mu as usize];

        for i in 0..weights.len() {
            weights[i] = (mu as f64 + 0.5f64).log2() - (i as f64 + 1f64).log2();
        }

        let weights = tch::Tensor::from_slice(weights.as_slice());
        let weights = weights.copy() / weights.sum(Some(tch::Kind::Float));
        let weights = weights.totype(tch::Kind::Float);

        let mut mueff = [0f64; 1];

        (weights.sum(Some(tch::Kind::Float)).pow_(2)
            / weights.copy().pow_(2).sum(Some(tch::Kind::Float)))
        .to_kind(tch::Kind::Double)
        .copy_data(&mut mueff, 1);

        let mueff = mueff[0];

        let cc = (4f64 + mueff / N as f64) / (N as f64 + 4f64 + 2f64 * mueff / N as f64);
        let cs = (mueff + 2f64) / (N as f64 + mueff + 5f64);
        let c1 = 2f64 / ((N as f64 + 1.3f64).powi(2) + mueff);
        let cmu = 2f64 * (mueff - 2f64 + 1f64 / mueff) / ((N as f64 + 2f64).powi(2) + mueff);

        let damps =
            1f64 + 2f64 * f64::max(0f64, ((mueff - 1f64) / (N as f64 + 1f64)).sqrt() - 1f64) + cs;

        let variation = tch::Tensor::zeros([N], (tch::Kind::Float, **device));

        let B = tch::Tensor::eye(N, (tch::Kind::Float, **device));
        let D = tch::Tensor::from_slice(vec![10f32.powi(-6); N as usize].as_slice())
            .diag(0)
            .to_device(**device);
        let C = tch::Tensor::matmul(&D, &D);
        let invsqrtC = tch::Tensor::from_slice(vec![10f32.powi(6); N as usize].as_slice())
            .diag(0)
            .to_device(**device);

        let Cold = tch::Tensor::eye(N, (tch::Kind::Float, **device));
        let Dinv = tch::Tensor::eye(N, (tch::Kind::Float, **device));

        let pc = tch::Tensor::zeros([N], (tch::Kind::Float, **device));
        let ps = tch::Tensor::zeros([N], (tch::Kind::Float, **device));

        let newpop =
            vec![std::rc::Rc::new(std::cell::RefCell::new(tch::Tensor::new())); lambda as usize];

        let counteval = 0;
        let eigeneval = 0;
        let gen = 0;

        Self {
            vs,

            cc,
            cs,
            c1,
            cmu,

            sigma,
            xmean,
            variation,
            newpop,
            N,
            lambda,
            mu,
            weights,
            mueff,
            damps,
            B,
            D,
            Dinv,
            C,
            Cold,
            invsqrtC,
            pc,
            ps,

            counteval,
            eigeneval,
            gen,
        }
    }
}

impl CMAES {
    fn vs_to_flattensor(vs: RefVs) -> tch::Tensor {
        let binding = vs.borrow().variables();
        let mut names_sorted = binding.keys().collect::<Vec<_>>();
        names_sorted.sort();

        let flatlist = names_sorted
            .iter()
            .map(|name| vs.borrow().variables().get(*name).unwrap().flatten(0, -1))
            .collect::<Vec<tch::Tensor>>();
        tch::Tensor::concat(&flatlist, 0)
    }

    fn flattensor_to_vs(layout: RefVs, tensor: tch::Tensor) -> tch::nn::VarStore {
        let newvs = tch::nn::VarStore::new(**device);

        let binding = layout.borrow().variables();
        let mut names_sorted = binding.keys().collect::<Vec<_>>();
        names_sorted.sort();

        let mut start_index = 0;
        for name in names_sorted {
            let layoutvar = binding.get(name).unwrap();
            let len = layoutvar.flatten(0, -1).size()[0];
            let val = tensor.i(start_index..start_index+len).unflatten(0, layoutvar.size());
            newvs.root().var(name, layoutvar.size().as_slice(), tch::nn::init::Init::Const(0f64)).copy_(&val);

            start_index += len;
        }

        newvs
    }
}

impl MilkshakeOptimizer for CMAES {
    fn ask(&mut self) -> Vec<RefVs> {

        for i in 0..self.lambda {
            let noise = tch::Tensor::randn([self.N], (tch::Kind::Float, **device));
            self.newpop[i as usize] = std::rc::Rc::new(std::cell::RefCell::new(
                &self.xmean + (self.sigma * &self.B * &self.D).mv(&noise),
            ));
        }

        let mut res = vec![];

        for candidate in &self.newpop {
            res.push(std::rc::Rc::new(std::cell::RefCell::new(
                Self::flattensor_to_vs(self.vs.clone(), candidate.borrow().copy()),
            )));
        }

        return res;
    }

    fn tell(&mut self, solutions: Vec<RefVs>, losses: Vec<tch::Tensor>) {
        self.counteval += self.lambda;

        let fitvals = tch::Tensor::stack(losses.as_slice(), 0).sort(0, false);

        let arIndexLocal = fitvals.1.copy().to_device(tch::Device::Cpu);

        let arindex = unsafe {
            std::slice::from_raw_parts(
                arIndexLocal.data_ptr() as *const i64,
                arIndexLocal.size()[0] as usize,
            )
        };

        let elite_indices = &arindex[0..self.mu as usize];

        let elite_solutions: Vec<tch::Tensor> = elite_indices
            .iter()
            .map(|i| self.newpop[*i as usize].borrow().copy())
            .collect();

        let meanold = self.xmean.copy();

        self.xmean = elite_solutions[0].copy() * self.weights.get(0);

        for i in 1..self.mu {
            self.xmean += elite_solutions[i as usize].copy() * self.weights.get(i);
        }

        let zscore = (&self.xmean - &meanold) / self.sigma;

        self.ps = (1f64 - self.cs) * &self.ps + (self.cs * (2f64 - self.cs) * self.mueff).sqrt() * &self.invsqrtC.mv(&zscore);

        let correlation = self.ps.norm().pow_tensor_scalar(2) / self.N / (1f64 - (1f64 - self.cs).powf(2f64 * self.counteval as f64 / self.lambda as f64));
        let hsig = unsafe { *(correlation.totype(tch::Kind::Float).to_device(tch::Device::Cpu).data_ptr() as *const f32) } < (2f64 + 4f64 / (self.N as f64 + 1f64)) as f32;
        let hsig = match hsig {
            true => { 1f64 }
            false => { 0f64 }
        };

        self.pc = (1f64 - self.cc) * &self.pc + hsig * (self.cc * (2. - self.cc) * self.mueff).sqrt() * &zscore;

        self.Cold = self.C.copy();
        self.C = (elite_solutions[0].copy() - &meanold) * (elite_solutions[0].copy() - &meanold).unsqueeze(1) * self.weights.get(0);

        for i in 1..self.mu {
            self.C += (elite_solutions[0].copy() - &meanold) * (elite_solutions[0].copy() - &meanold).unsqueeze(1) * self.weights.get(i);
        }

        self.C /= self.sigma.powi(2);
        self.C = (1f64 - self.c1 - self.cmu) * &self.Cold + self.cmu * &self.C + self.c1 * ((&self.pc * &self.pc.copy().unsqueeze(1)) + (1f64 - hsig) * self.cc * (2f64 - self.cc) * &self.Cold);

        if (self.counteval - self.eigeneval) as f64 > self.lambda as f64 / (self.c1 + self.cmu) / self.N as f64 / 10f64 {
            self.eigeneval = self.counteval;

            let db = self.C.linalg_eigh("L");

            db.0.print();
            db.1.print();

            self.D = db.0.sqrt().diag_embed(0, -2, -1);
            self.B = db.1.copy();
            self.Dinv = self.D.pow_(-1f64);
            self.invsqrtC = &self.B * &self.Dinv * self.B.copy().t_();
        }
    }

    fn result(&mut self) -> RefVs {
        std::rc::Rc::new(std::cell::RefCell::new(Self::flattensor_to_vs(
            self.vs.clone(),
            self.xmean.copy(),
        )))
    }

    fn grads(&mut self) -> bool {
        false
    }
}
