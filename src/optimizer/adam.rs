use crate::optimizer::MilkshakeOptimizer;
use std::ops::Deref;

pub struct ADAM {
    pub vs: std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>,
    pub opt: tch::nn::Optimizer,
}

impl ADAM {
    pub fn new(lr: f64, vs: std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>) -> Self {
        let opt =
            tch::nn::OptimizerConfig::build(tch::nn::Adam::default(), vs.borrow().deref(), lr)
                .expect("Failed to construct Adam Optimizer");

        Self { vs, opt }
    }
}

impl MilkshakeOptimizer for ADAM {
    fn ask(&mut self) -> Vec<std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>> {
        vec![self.vs.clone()]
    }

    fn tell(
        &mut self,
        solutions: Vec<std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>>,
        losses: Vec<tch::Tensor>,
    ) {
        assert_eq!(solutions.len(), 1);
        assert_eq!(losses.len(), 1);
        assert!(std::rc::Rc::ptr_eq(solutions.first().unwrap(), &self.vs));

        self.opt.zero_grad();
        losses.first().unwrap().backward();
        self.opt.step();
    }

    fn result(&mut self) -> std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>> {
        self.vs.clone()
    }

    fn grads(&mut self) -> bool {
        true
    }
}
