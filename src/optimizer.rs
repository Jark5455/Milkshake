pub mod adam;
pub mod cmaes;

extern crate tch;

pub type RefVs = std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>;

pub trait MilkshakeOptimizer {
    fn ask(&mut self) -> Vec<RefVs>;
    fn tell(&mut self, solutions: Vec<RefVs>, losses: Vec<tch::Tensor>);
    fn result(&mut self) -> RefVs;

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
