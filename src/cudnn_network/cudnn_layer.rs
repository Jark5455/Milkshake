// The following code is stolen and reinterpreted from cudnn mnist sample

use std::rc::Rc;
use rcudnn::TensorDescriptor;
use crate::cudnn_network::blob::Blob;

trait CUDNNLayer {
    fn name_(&mut self) -> &mut String;
    fn input_desc_(&mut self) -> Rc<Option<TensorDescriptor>>;
    fn output_desc_(&mut self) -> Rc<Option<TensorDescriptor>>;
    fn filter_desc_(&mut self) -> Rc<Option<TensorDescriptor>>;
    fn bias_desc_(&mut self) -> Rc<Option<TensorDescriptor>>;

    fn input_(&mut self) -> Rc<Option<Blob<f32>>>;
    fn output_(&mut self) -> Rc<Option<Blob<f32>>>;
    fn grad_input_(&mut self) -> Rc<Option<Blob<f32>>>;
    fn grad_output_(&mut self) -> Rc<Option<Blob<f32>>>;
    fn freeze_(&mut self) -> &mut bool;
    fn weights_(&mut self) -> Rc<Option<Blob<f32>>>;
    fn biases_(&mut self) -> Rc<Option<Blob<f32>>>;
    fn grad_weights_(&mut self) -> Rc<Option<Blob<f32>>>;
    fn grad_biases_(&mut self) -> Rc<Option<Blob<f32>>>;
    fn batch_size_(&mut self) -> &mut u32;

    fn init_weight_bias_(seed: u32);
    fn update_weights_biases_(learning_rate: f32);
    fn load_pretrain_(&mut self) -> &mut bool;
    fn load_parameter_(&self) -> u32;
    fn save_parameter_(&self) -> u32;
    fn gradient_stop_(&mut self) -> &mut bool;

    fn forward(&mut self, input: Box<Blob<f32>>) -> Rc<Blob<f32>>;
    fn backward(&mut self, grad_input: Box<Blob<f32>>) -> Rc<Blob<f32>>;
    fn loss(&mut self, target: Box<Blob<f32>>) -> f32;
    fn accuracy(&mut self, target: Box<Blob<f32>>) -> u32;
    fn set_load_pretrain(&mut self) {
        *self.load_pretrain_() = true;
    }
    fn set_gradient_stop(&mut self) {
        *self.gradient_stop_() = true;
    }
    fn freeze_layer(&mut self) {
        *self.freeze_() = true;
    }
    fn unfreeze_layer(&mut self) {
        *self.freeze_() = false;
    }
}

struct CUDNNDense {
    pub name: String,
    pub input_desc: Option<TensorDescriptor>,
    pub output_desc: Option<TensorDescriptor>,
    pub filter_desc: Option<TensorDescriptor>,
    pub bias_desc: Option<TensorDescriptor>,
    pub input: Option<Blob<f32>>,
    pub output: Option<Blob<f32>>,
    pub grad_input: Option<Blob<f32>>,
    pub grad_output: Option<Blob<f32>>,
    pub freeze: bool,
    pub weights: Option<Blob<f32>>,
    pub biases: Option<Blob<f32>>,
    pub grad_weights: Option<Blob<f32>>,
    pub grad_biases: Option<Blob<f32>>,
    pub batch_size: u32,
    pub load_pretrain: bool,
    pub load_parameter: u32,
    pub save_parameter: u32,
    pub grad_stop: bool
}

impl CUDNNLayer for CUDNNDense {
    fn name_(&mut self) -> &mut String {
        todo!()
    }

    fn input_desc_(&mut self) -> Rc<Option<TensorDescriptor>> {
        todo!()
    }

    fn output_desc_(&mut self) -> Rc<Option<TensorDescriptor>> {
        todo!()
    }

    fn filter_desc_(&mut self) -> Rc<Option<TensorDescriptor>> {
        todo!()
    }

    fn bias_desc_(&mut self) -> Rc<Option<TensorDescriptor>> {
        todo!()
    }

    fn input_(&mut self) -> Rc<Option<Blob<f32>>> {
        todo!()
    }

    fn output_(&mut self) -> Rc<Option<Blob<f32>>> {
        todo!()
    }

    fn grad_input_(&mut self) -> Rc<Option<Blob<f32>>> {
        todo!()
    }

    fn grad_output_(&mut self) -> Rc<Option<Blob<f32>>> {
        todo!()
    }

    fn freeze_(&mut self) -> &mut bool {
        todo!()
    }

    fn weights_(&mut self) -> Rc<Option<Blob<f32>>> {
        todo!()
    }

    fn biases_(&mut self) -> Rc<Option<Blob<f32>>> {
        todo!()
    }

    fn grad_weights_(&mut self) -> Rc<Option<Blob<f32>>> {
        todo!()
    }

    fn grad_biases_(&mut self) -> Rc<Option<Blob<f32>>> {
        todo!()
    }

    fn batch_size_(&mut self) -> &mut u32 {
        todo!()
    }

    fn init_weight_bias_(seed: u32) {
        todo!()
    }

    fn update_weights_biases_(learning_rate: f32) {
        todo!()
    }

    fn load_pretrain_(&mut self) -> &mut bool {
        todo!()
    }

    fn load_parameter_(&self) -> u32 {
        todo!()
    }

    fn save_parameter_(&self) -> u32 {
        todo!()
    }

    fn gradient_stop_(&mut self) -> &mut bool {
        todo!()
    }

    fn forward(&mut self, input: Box<Blob<f32>>) -> Rc<Blob<f32>> {
        todo!()
    }

    fn backward(&mut self, grad_input: Box<Blob<f32>>) -> Rc<Blob<f32>> {
        todo!()
    }

    fn loss(&mut self, target: Box<Blob<f32>>) -> f32 {
        todo!()
    }

    fn accuracy(&mut self, target: Box<Blob<f32>>) -> u32 {
        todo!()
    }
}