use crate::device;
use crate::optimizer::MilkshakeOptimizer;

pub struct ADAM<'opt> {
    pub vs: &'opt tch::nn::VarStore,
    pub opt: tch::nn::Optimizer,
}

impl<'opt> ADAM<'opt> {
    pub fn new(lr: f64, vs: &tch::nn::VarStore) -> Box<dyn MilkshakeOptimizer> {
        let opt = tch::nn::OptimizerConfig::build(tch::nn::Adam::default(), vs, lr).expect("Failed to construct Adam Optimizer");

        return Box::new(Self {
            vs,
            opt
        });
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