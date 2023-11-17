use crate::optimizer::MilkshakeOptimizer;

pub struct ADAM {
    pub vs: std::sync::Arc<tch::nn::VarStore>,
    pub opt: tch::nn::Optimizer,
}

impl ADAM {
    pub fn new(lr: f64, vs: std::sync::Arc<tch::nn::VarStore>) -> Self {
        let opt = tch::nn::OptimizerConfig::build(tch::nn::Adam::default(), vs.as_ref(), lr)
            .expect("Failed to construct Adam Optimizer");

        Self { vs, opt }
    }
}

impl MilkshakeOptimizer for ADAM {
    fn ask(&mut self) {
        // nothing much to do here
    }

    fn tell(&mut self, loss: tch::Tensor) {
        // updates varstore automatically

        self.opt.zero_grad();
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
