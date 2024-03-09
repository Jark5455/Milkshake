use crate::device;
use crate::optimizer::MilkshakeOptimizer;
use crate::optimizer::RefVs;

use tch::IndexOp;

// This is a barebones CMAES implementation in pytorch

pub struct CMAES {
    pub vs: RefVs,
    pub xmean: tch::Tensor,
    pub z: tch::Tensor,
    pub s: tch::Tensor,
    pub N: i64,
    pub sigma: f64,
    pub lambda: i64,
    pub mu: i64,
    pub weights: tch::Tensor,
    pub mueff: f64,
    pub cc: f64,
    pub cs: f64,
    pub c1: f64,
    pub cmu: f64,
    pub damps: f64,
    pub chiN: f64,
    pub B: tch::Tensor,
    pub D: tch::Tensor,
    pub C: tch::Tensor,
    pub pc: tch::Tensor,
    pub ps: tch::Tensor,
    pub gen: i64,
}

impl CMAES {
    pub fn new(vs: RefVs, sigma: Option<f64>, popsize: Option<i64>) -> Self {
        let sigma = sigma.unwrap_or(0.5f64);

        let xmean = Self::vs_to_flattensor(vs.clone());

        let N = xmean.size()[0];

        let lambda = popsize.unwrap_or(4 + (3f64 * (N as f64).ln()).floor() as i64);
        let mu = lambda as f64 / 2f64;

        let weights = tch::Tensor::from_slice(&[(mu + 0.5f64).log2()]).to_device(**device)
            - tch::Tensor::linspace(1f64, mu, mu.floor() as i64, (tch::Kind::Float, **device));
        let weights = weights.copy() / weights.sum(Some(tch::Kind::Float));
        let weights = weights.copy() / weights.sum(Some(tch::Kind::Float));
        let weights = weights.totype(tch::Kind::Float);

        let mu = mu.floor() as i64;

        let mut mueff = [0f64; 1];

        (weights.sum(Some(tch::Kind::Float)).pow_(2)
            / weights.copy().pow_(2).sum(Some(tch::Kind::Float)))
        .to_kind(tch::Kind::Double)
        .copy_data(&mut mueff, 1);

        let mueff = mueff[0];

        let cc = (4f64 + mueff / N as f64) / (N as f64 + 4f64 + 2f64 * mueff / N as f64);
        let cs = (mueff + 2f64) / (N as f64 + mueff + 5f64);
        let c1 = 2f64 / ((N as f64 + 1.3f64).powi(2) + mueff);
        let cmu = f64::min(
            1f64 - c1,
            2f64 * (mueff - 2f64 + 1f64 / mueff) / ((N as f64 + 2f64).powi(2) + mueff),
        );

        let damps = f64::min(0f64, ((mu as f64 - 1f64) / (N as f64 + 1f64)).sqrt() - 1f64) + cs;

        let chiN = (N as f64).sqrt()
            * (1f64 - 1f64 / (4f64 * N as f64) + 1f64 / (21f64 * (N as f64).sqrt()));

        let B = tch::Tensor::eye(N, (tch::Kind::Float, **device));
        let D = tch::Tensor::eye(N, (tch::Kind::Float, **device));
        let C = tch::Tensor::matmul(&B.matmul(&D), &B.matmul(&D).t_());

        let z = tch::Tensor::randn([N, lambda], (tch::Kind::Float, **device));
        let s = xmean.view([-1, 1]) + sigma * B.matmul(&D.matmul(&z));

        let pc = tch::Tensor::zeros([N], (tch::Kind::Float, **device));
        let ps = tch::Tensor::zeros([N], (tch::Kind::Float, **device));

        let gen = 0;

        Self {
            vs,
            N,
            z,
            s,
            xmean,
            sigma,
            lambda,
            mu,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chiN,
            pc,
            ps,
            B,
            D,
            C,
            gen,
        }
    }
}

impl CMAES {
    fn vs_to_flattensor(vs: RefVs) -> tch::Tensor {
        let flatlist: Vec<tch::Tensor> = vs
            .borrow()
            .trainable_variables()
            .iter()
            .map(|var| var.flatten(0, (var.dim() - 1) as i64))
            .collect();
        tch::Tensor::concat(&flatlist, 0)
    }

    fn flattensor_to_vs(layout: RefVs, tensor: tch::Tensor) -> tch::nn::VarStore {
        let newvs = tch::nn::VarStore::new(**device);

        for (name, tensor) in &layout.borrow().variables_.lock().unwrap().named_variables {
            newvs
                .root()
                .var(
                    &*name,
                    tensor.size().as_slice(),
                    tch::nn::init::Init::Const(0f64),
                )
                .copy_(tensor);
        }

        let mut startindex = 0;
        for mut var in newvs.trainable_variables() {
            let len = var.flatten(0, -1).size()[0];

            let val = tensor
                .i(startindex..startindex + len)
                .unflatten(0, var.size());
            var.copy_(&val);

            startindex = startindex + len;
        }

        newvs
    }
}

impl MilkshakeOptimizer for CMAES {
    fn ask(&mut self) -> Vec<RefVs> {
        let mut z = tch::Tensor::randn([self.N, self.lambda], (tch::Kind::Float, **device));
        let mut s = self.xmean.view([-1, 1]) + self.sigma * self.B.matmul(&self.D.matmul(&z));

        self.z = z.t_();
        self.s = s.t_();

        let candidates = tch::Tensor::unbind(&self.s, 0);

        let mut res = vec![];

        for candidate in candidates {
            res.push(std::rc::Rc::new(std::cell::RefCell::new(
                Self::flattensor_to_vs(self.vs.clone(), candidate),
            )));
        }

        return res;
    }

    fn tell(&mut self, solutions: Vec<RefVs>, losses: Vec<tch::Tensor>) {
        let fitvals = tch::Tensor::stack(losses.as_slice(), 0).sort(0, false);

        let arIndexLocal = fitvals.1.copy().to_device(tch::Device::Cpu);

        let arindex = unsafe {
            std::slice::from_raw_parts(
                arIndexLocal.data_ptr() as *const i64,
                arIndexLocal.size()[0] as usize,
            )
        };

        let elite_indices = &arindex[0..self.mu as usize];
        let elite_solutions: Vec<RefVs> = elite_indices
            .iter()
            .map(|i| solutions[*i as usize].clone())
            .collect();

        let z = self.z.index_select(
            0,
            &tch::Tensor::from_slice(elite_indices).to_device(**device),
        );

        let g = tch::Tensor::stack(
            elite_solutions
                .iter()
                .map(|s| Self::vs_to_flattensor(s.clone()))
                .collect::<Vec<tch::Tensor>>()
                .as_slice(),
            0,
        );

        self.xmean = (g * self.weights.unsqueeze(1)).sum_dim_intlist(
            &[0i64][..],
            false,
            Some(tch::Kind::Float),
        );

        let vs = Self::flattensor_to_vs(self.vs.clone(), self.xmean.copy());
        self.vs
            .borrow_mut()
            .copy(&vs)
            .expect("Failed to update real vs");

        let zmean = (z.copy() * self.weights.unsqueeze(1)).sum_dim_intlist(
            &[0i64][..],
            false,
            Some(tch::Kind::Float),
        );

        self.ps = (1f64 - self.cs) * &self.ps
            + (self.cs * (2.0 - self.cs)).sqrt() * self.B.matmul(&zmean);

        let correlation = self.ps.norm() / self.chiN;

        let correlation = unsafe {
            std::slice::from_raw_parts(
                correlation
                    .to_kind(tch::Kind::Double)
                    .to_device(tch::Device::Cpu)
                    .data_ptr() as *const f64,
                1,
            )[0]
        };

        let denominator =
            (1f64 - (1f64 - self.cs).powf(2f64 * self.gen as f64 / self.lambda as f64)).sqrt();
        let threshold = 140f64 / self.N as f64 + 1f64;

        let hsig = match correlation / denominator < threshold {
            true => 1f64,
            false => 0f64,
        };

        self.sigma = self.sigma * ((self.cs / self.damps) * (correlation - 1.0)).exp();

        self.pc = (1f64 - self.cc) * &self.pc
            + hsig
                * (self.cc * (2f64 - self.cc) * self.mueff).sqrt()
                * self.B.matmul(&self.D).matmul(&zmean);

        let pc_cov = self.pc.unsqueeze(1).matmul(&self.pc.unsqueeze(1).t_());
        let pc_cov = pc_cov + (1f64 - hsig) * self.cc * (2f64 - self.cc) * &self.C;

        let bdz = self.B.matmul(&self.D).matmul(&z.copy().t_());
        let cmu_cov = tch::Tensor::matmul(&bdz, &self.weights.diag_embed(0, -2, -1));
        let cmu_cov = cmu_cov.matmul(&bdz.copy().t_());

        self.C = (1.0 - self.c1 - self.cmu) * &self.C + (self.c1 * pc_cov) + (self.cmu * cmu_cov);

        let eig = self.C.linalg_eigh("L");

        self.D = eig.0;
        self.C = eig.1;
        self.D = self.D.sqrt().diag_embed(0, -2, -1);

        self.gen += 1;
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
