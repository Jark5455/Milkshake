pub mod adam;
pub mod cmaes;

extern crate tch;

pub trait MilkshakeOptimizer {
    fn ask(&mut self) -> Vec<std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>>;
    fn tell(
        &mut self,
        solutions: Vec<std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>>,
        losses: Vec<tch::Tensor>,
    );
    fn result(&mut self) -> std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>;

    fn disp(&mut self) {
        println!(
            "Display not implemented for this optimizer: {}",
            std::any::type_name::<Self>()
        )
    }

    fn grads(&mut self) -> bool {
        false
    }
}
