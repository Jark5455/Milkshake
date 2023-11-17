pub mod adam;

extern crate tch;

pub trait MilkshakeOptimizer {
    fn ask(&mut self);
    fn tell(&mut self, loss: tch::Tensor);
    fn stop(&mut self);
    fn result(&mut self);

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
