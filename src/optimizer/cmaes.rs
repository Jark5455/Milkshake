use crate::device;
use crate::optimizer::MilkshakeOptimizer;
use crate::optimizer::RefVs;

use tch::IndexOp;

// This is a barebones CMAES implementation in pytorch

pub struct CMAES {
    pub vs: RefVs,
    pub xmean: tch::Tensor,

    pub cc: f64,
    pub cs: f64,
    pub c1: f64,
    pub cmu: f64,

    pub z: tch::Tensor,
    pub s: tch::Tensor,

    pub N: i64,
    pub lambda: i64,
    pub mu: i64,

    pub sigma: f64,
    pub weights: tch::Tensor,
    pub mueff: f64,
    pub damps: f64,
    pub chiN: f64,
    pub B: tch::Tensor,
    pub D: tch::Tensor,
    pub C: tch::Tensor,
    pub pc: tch::Tensor,
    pub ps: tch::Tensor,

    pub eigeneval: i64,
    pub counteval: i64,
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
        let weights = weights.totype(tch::Kind::Float).to_device(**device);

        let mueff = unsafe {
            *((weights.sum(Some(tch::Kind::Float)).pow_(2)
                / weights.copy().pow_(2).sum(Some(tch::Kind::Float)))
                .to_kind(tch::Kind::Double)
                .to_device(tch::Device::Cpu)
                .data_ptr() as *const f64)
        };

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

        let z = tch::Tensor::zeros([lambda, N], (tch::Kind::Float, **device));
        let s = tch::Tensor::zeros([lambda, N], (tch::Kind::Float, **device));

        let pc = tch::Tensor::zeros([N], (tch::Kind::Float, **device));
        let ps = tch::Tensor::zeros([N], (tch::Kind::Float, **device));

        let eigeneval = 0;
        let counteval = 0;

        Self {
            vs,
            xmean,

            cc,
            cs,
            c1,
            cmu,

            z,
            s,

            N,
            lambda,
            mu,

            sigma,
            weights,
            mueff,
            damps,
            chiN,

            B,
            D,
            C,
            pc,
            ps,

            eigeneval,
            counteval
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
            let val = tensor
                .i(start_index..start_index + len)
                .unflatten(0, layoutvar.size());
            newvs
                .root()
                .var(
                    name,
                    layoutvar.size().as_slice(),
                    tch::nn::init::Init::Const(0f64),
                )
                .copy_(&val);

            start_index += len;
        }

        newvs
    }
}

impl MilkshakeOptimizer for CMAES {
    fn ask(&mut self) -> Vec<RefVs> {
        self.z = tch::Tensor::randn([self.N, self.lambda], (tch::Kind::Float, **device));
        self.s = self.xmean.view([-1, 1]) + self.sigma * self.B.matmul(&self.D.matmul(&self.z));

        let candidates = tch::Tensor::unbind(&self.s.copy().t_(), 0);

        let mut res = vec![];

        for candidate in candidates {
            res.push(std::rc::Rc::new(std::cell::RefCell::new(
                Self::flattensor_to_vs(self.vs.clone(), candidate),
            )));
        }

        return res;
    }

    fn tell(&mut self, solutions: Vec<RefVs>, losses: Vec<tch::Tensor>) {

        self.counteval += self.lambda;

        // select elite solutions and their respective z scores

        let fitvals = tch::Tensor::stack(losses.as_slice(), 0).sort(0, false);

        let arIndexLocal = fitvals.1.copy().to_device(tch::Device::Cpu);

        let arindex = unsafe {
            std::slice::from_raw_parts(
                arIndexLocal.data_ptr() as *const i64,
                arIndexLocal.size()[0] as usize,
            )
        };

        let elite_indices = &arindex[0..self.mu as usize];
        let elite_solutions = elite_indices
            .iter()
            .map(|i| Self::vs_to_flattensor(solutions[*i as usize].clone()))
            .collect::<Vec<tch::Tensor>>();

        let z = self.z.index_select(
            1,
            &tch::Tensor::from_slice(elite_indices).to_device(**device),
        );

        // recombination

        let recombination_solutions = tch::Tensor::stack(elite_solutions.as_slice(), 0);

        self.xmean = (recombination_solutions * self.weights.unsqueeze(1)).sum_dim_intlist(
            &[0i64][..],
            false,
            Some(tch::Kind::Float),
        );

        let zmean = (z.copy().t_() * self.weights.unsqueeze(1)).sum_dim_intlist(
            &[0i64][..],
            false,
            Some(tch::Kind::Float),
        );

        self.ps = (1f64 - self.cs) * &self.ps + (self.cs * (2.0 - self.cs) * self.mueff).sqrt() * &self.B.mv(&zmean);
        let psNorm = unsafe { *(self.ps.norm().to_device(tch::Device::Cpu).data_ptr() as *const f32) } as f64;

        let hsig = psNorm / (1f64 - (1f64 - self.cs).powi((2 * self.counteval / self.lambda) as i32)).sqrt() / self.chiN < (1.4 + 2f64 / (self.N + 1) as f64);

        let hsig = match hsig {
            true => {1f64}
            false => {0f64}
        };

        self.pc = (1f64 - self.cc) * &self.pc
            + hsig
            * (self.cc * (2f64 - self.cc) * self.mueff).sqrt()
            * self.B.matmul(&self.D).matmul(&zmean);

        let bdz = self.B.matmul(&self.D).matmul(&z);

        self.C = (1f64 - self.c1 - self.cmu) * &self.C
            + self.c1 * (self.pc.unsqueeze(1).matmul(&self.pc.unsqueeze(1).t_())
                + (1f64 - hsig) * self.cc * (2f64 - self.cc) * &self.C)
            + self.cmu
                * &bdz.matmul(&self.weights.diag_embed(0, -2, -1).matmul(&bdz.copy().t_()));


        self.sigma = self.sigma * ((self.cs / self.damps) * (psNorm / self.chiN - 1f64)).exp();

        if (self.counteval - self.eigeneval) as f64
            > self.lambda as f64 / (self.c1 + self.cmu) / self.N as f64 / 10f64
        {
            self.eigeneval = self.counteval;
            self.C = (&self.C + &self.C.copy().t_()) / 2f64; // enforce symmetry (even though pytorch doesn't check symmetry)

            let db = self.C.linalg_eigh("L");

            self.D = db.0.copy().sqrt().diag_embed(0, -2, -1);
            self.B = db.1.copy();
        }

        let fit1 = unsafe { *(fitvals.0.select(0, 0).to_device(tch::Device::Cpu).data_ptr() as *const f32) };
        let fit2 = unsafe { *(fitvals.0.select(0, (0.7 * self.lambda as f64).ceil() as i64).to_device(tch::Device::Cpu).data_ptr() as *const f32) };

        if fit1 == fit2 {
            self.sigma = self.sigma * (0.2f64 + self.cs/self.damps).exp();
            println!("WARNING: flat fitness detected, adjusting step size");
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
