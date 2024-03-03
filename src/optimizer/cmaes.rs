use crate::device;
use crate::optimizer::MilkshakeOptimizer;
use crate::optimizer::RefVs;

// This is an implementation of barecmaes2.py from http://www.cmap.polytechnique.fr/~nikolaus.hansen/barecmaes2.py
// This is an implementation of purecma.py from https://github.com/CMA-ES/pycma/blob/development/cma/purecma.py

/*

struct RecombinationWeights {
    pub weights: Vec<f64>,
    pub exponent: f64,
    pub mu: i32,
    pub mueff: f64,
}

union RecombinationWeightsLenData {
    v1: std::mem::ManuallyDrop<Vec<f64>>,
    u2: usize,
}

enum RecombinationWeightsLenKind {
    Vector,
    Size,
}

struct RecombinationWeightsLen {
    data: RecombinationWeightsLenData,
    kind: RecombinationWeightsLenKind,
}

impl RecombinationWeights {
    pub fn new(len: RecombinationWeightsLen, exponent: Option<f64>) -> Self {
        let exponent = exponent.unwrap_or(1f64);

        let mut weights = unsafe {
            match len.kind {
                RecombinationWeightsLenKind::Vector => {
                    std::mem::ManuallyDrop::<Vec<f64>>::into_inner(len.data.v1)
                }

                RecombinationWeightsLenKind::Size => {
                    let signed_power = |x: f64, expo: f64| -> f64 {
                        if expo == 1f64 {
                            return x;
                        }

                        let s = (x != 0f64) as i32 as f64 * x.signum();
                        s * x.abs().powf(expo)
                    };

                    let mut vec = vec![0f64; len.data.u2];

                    for i in 0..vec.len() {
                        vec[i] = signed_power(
                            ((len.data.u2 + 1) as f64 / 2f64).ln() - (i as f64 + 1f64).ln(),
                            exponent,
                        );
                    }

                    vec
                }
            }
        };

        assert!(weights.len() > 0);
        assert!(weights[0] > 0f64);
        assert!(weights[weights.len() - 1] > 0f64);

        for i in 0..weights.len() - 1 {
            assert!(weights[i] > weights[i + 1]);
        }

        let mut mu = 0i32;
        for i in &weights {
            if *i > 0f64 {
                mu += 1
            }
        }

        let mut spos = 0f64;
        for i in 0..mu {
            spos += weights[i as usize];
        }

        assert!(spos >= 0f64);

        for i in 0..weights.len() {
            weights[i] = weights[i] / spos;
        }

        let mut wsquared = 0f64;
        for i in &weights {
            wsquared = wsquared + i.powi(2);
        }

        let mueff = 1f64 / wsquared;

        let mut sneg = 0f64;
        for i in mu..weights.len() as i32 {
            sneg = sneg + weights[i as usize];
        }

        let wsum: f64 = weights.iter().sum();
        assert!((sneg - wsum).powi(2) < 10f64.powi(-11));

        RecombinationWeights {
            weights,
            exponent,
            mu,
            mueff,
        }
    }
}

impl<Idx> std::ops::Index<Idx> for RecombinationWeights
where
    Idx: std::slice::SliceIndex<[f64], Output = f64>,
{
    type Output = f64;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.weights[index]
    }
}

impl<Idx> std::ops::IndexMut<Idx> for RecombinationWeights
where
    Idx: std::slice::SliceIndex<[f64], Output = f64>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.weights[index]
    }
}

pub struct CMAESParameters {
    pub N: u32,
    pub chiN: f64,
    pub lambda: u32,
    pub mu: u32,
    pub weights: RecombinationWeights,
    pub mueff: f64,
    pub cc: f64,
    pub cs: f64,
    pub ci: f64,
    pub cmu: f64,
    pub damps: f64,
}

impl CMAESParameters {
    pub fn new(N: u32, popsize: Option<u32>) -> Self {
        let chiN = (N as f64).powf(0.5f64)
            * (1f64 - 1f64 / (4f64 * N as f64) + 1f64 / (21f64 * (N as f64).powi(2)));
        let lambda = match popsize {
            None => 4 + (3 * (N as f64).log2() as u32),
            Some(lam) => lam,
        };

        let mu = lambda / 2;

        let weights = RecombinationWeights::new(
            RecombinationWeightsLen {
                data: RecombinationWeightsLenData {
                    u2: lambda as usize,
                },
                kind: RecombinationWeightsLenKind::Size,
            },
            None,
        );

        let mueff = weights.mueff;
        let cc = (4f64 + mueff / N as f64) / (N as f64 + 4f64 + 2f64 * mueff / N as f64);
        let cs = (mueff + 2f64) / (N as f64 + mueff + 5f64);
        let ci = 2f64 / ((N as f64 + 1.3).powi(2) + mueff);
        let cmu = f64::min(
            1f64 - ci,
            2f64 * (mueff - 2f64 + 1f64 / mueff) / ((N as f64 + 2f64).powi(2) + mueff),
        );
        let damps = 2f64 * mueff / (lambda as f64) + 0.3f64 + cs;

        Self {
            N,
            chiN,
            lambda,
            mu,
            weights,
            mueff,
            cc,
            cs,
            ci,
            cmu,
            damps,
        }
    }
}

pub struct CMAES {
    pub xmean: RefVs,
}

impl CMAES {
    pub fn new(
        xstart: RefVs,
        sigma: f64,
        popsize: Option<u32>,
        maxfevals: Option<u32>,
        ftarget: Option<f64>,
    ) {
        let ftarget = ftarget.unwrap_or(0f64);
        let N = xstart.borrow().trainable_variables().len() as u32;
        let parameters = CMAESParameters::new(N, popsize);

        let maxfevals = match maxfevals {
            None => {
                100 * parameters.lambda
                    + 150 * (N + 3).pow(2) * (parameters.lambda as f64).sqrt() as u32
            }
            Some(maxevals) => maxevals,
        };

        let xmean = std::rc::Rc::new(std::cell::RefCell::new(tch::nn::VarStore::new(**device)));
        xmean.copy(xstart.borrow());

        let pc = tch::Tensor::from_slice(vec![0; N as usize].as_slice()).to_device(**device);
        let ps = tch::Tensor::from_slice(vec![0; N as usize].as_slice()).to_device(**device);

        let B = tch::Tensor::eye(N as i64, (tch::Kind::Float, **device));
        let D = tch::Tensor::from_slice(vec![1; N as usize].as_slice()).to_device(**device);
        let C = tch::Tensor::eye(N as i64, (tch::Kind::Float, **device));
        let invSqrtC = tch::Tensor::eye(N as i64, (tch::Kind::Float, **device));
    }
}

*/

struct BestSolution {
    pub x: Option<RefVs>,
    pub f: Option<tch::Tensor>,
    pub evals: Option<u32>,
}

impl BestSolution {
    pub fn new(x: Option<RefVs>, f: Option<tch::Tensor>, evals: Option<u32>) -> Self {
        Self { x, f, evals }
    }

    pub fn update(&mut self, arx: Vec<RefVs>, arf: Vec<tch::Tensor>, evals: Option<u32>) {
        let arf = tch::Tensor::concat(arf.as_slice(), 9);

        if self.f == None
            || unsafe {
                *(arf
                    .min()
                    .less_equal_tensor(self.f.as_ref().unwrap())
                    .data_ptr() as *mut bool)
            }
        {
            let i = arf.argmin(None, false).f_int64_value(&[0]).unwrap();

            self.x = arx[i];
            self.f = arf[i];

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
    fn ask(&mut self) -> Vec<RefVs> {
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

            let z = self.D.copy().normal_(0f64, 1f64) * self.D.copy();
        }

        res
    }

    fn tell(&mut self, solutions: Vec<RefVs>, losses: Vec<tch::Tensor>) {
        self.counteval += losses.len() as u32;
        let N = solutions.len();
        let mut iN = Vec::new();
        for i in 0..N {
            iN.push(i);
        }

        let mut xold = tch::nn::VarStore::new(**device);
        xold.copy(&self.xmean).expect("Failed to copy xmean");

        let fitvals = tch::Tensor::cat(losses.as_slice(), 1).sort(0, false);

        self.fitvals = fitvals.0;

        let arindex = unsafe {
            std::slice::from_raw_parts(
                fitvals.1.data_ptr() as *mut i64,
                fitvals.1.size()[0] as usize,
            )
        };
        let mut arx = Vec::new();

        for i in arindex {
            arx.push(solutions[*i as usize].clone());
        }

        self.bestsolution
            .update(vec![arx[0]], vec![self.fitvals[0]], Some(self.counteval));
    }

    fn result(&mut self) -> RefVs {
        todo!()
    }
}
