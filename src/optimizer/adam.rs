use crate::optimizer::MilkshakeOptimizer;

pub struct ADAM {
    pub opt: tch::nn::Optimizer,
}

impl ADAM {
    pub fn new(lr: f64, vs: &tch::nn::VarStore) -> Self {
        let opt = tch::nn::OptimizerConfig::build(tch::nn::Adam::default(), vs, lr)
            .expect("Failed to construct Adam Optimizer");

        Self { opt }
    }
}

impl MilkshakeOptimizer for ADAM {
    fn ask(&mut self) {
        self.opt.zero_grad();
    }

    fn tell(&mut self, loss: tch::Tensor) {
        loss.backward();
        self.opt.step();
    }

    fn stop(&mut self) {
        // nothing much to do here
    }

    fn result(&mut self) {
        todo!()
    }

    fn grads(&mut self) -> bool {
        true
    }
}
