// The following code is stolen and reinterpreted from cudnn mnist sample

use std::ops::Deref;
use std::rc::Rc;
use cust::memory::DeviceBuffer;
use rand::distributions::Uniform;
use rand::prelude::{StdRng};
use rand::{Rng, SeedableRng};
use rcudnn::{cudaDeviceSynchronize, TensorDescriptor};
use rcudnn::cudaError_t::cudaSuccess;
use crate::cublas_handle_s;
use crate::cudnn_network::blob::Blob;
use crate::cudnn_network::blob::DeviceType::cuda;
use crate::cudnn_network::DEBUG_UPDATE;

trait CUDNNLayer {
    fn name_(&mut self) -> &mut String;
    fn input_desc_(&mut self) -> &mut Option<TensorDescriptor>;
    fn output_desc_(&mut self) -> &mut Option<TensorDescriptor>;
    fn filter_desc_(&mut self) -> &mut Option<TensorDescriptor>;
    fn bias_desc_(&mut self) -> &mut Option<TensorDescriptor>;

    fn input_(&mut self) -> &mut Option<Blob<f32>>;
    fn output_(&mut self) -> &mut Option<Blob<f32>>;
    fn grad_input_(&mut self) -> &mut Option<Blob<f32>>;
    fn grad_output_(&mut self) -> &mut Option<Blob<f32>>;
    fn freeze_(&mut self) -> &mut bool;
    fn weights_(&mut self) -> &mut Option<Blob<f32>>;
    fn biases_(&mut self) -> &mut Option<Blob<f32>>;
    fn grad_weights_(&mut self) -> &mut Option<Blob<f32>>;
    fn grad_biases_(&mut self) -> &mut Option<Blob<f32>>;
    fn gradient_stop_(&mut self) -> &mut bool;
    fn batch_size_(&mut self) -> &mut u32;
    fn load_pretrain_(&mut self) -> &mut bool;

    fn init_weight_bias(&mut self, seed: u32);
    fn update_weights_biases(&mut self, learning_rate: f32);
    fn load_parameter(&self) -> u32;
    fn save_parameter(&self) -> u32;

    fn forward(&mut self, input: Box<Blob<f32>>) -> &mut Blob<f32>;
    fn backward(&mut self, grad_input: Box<Blob<f32>>) -> &mut Blob<f32>;
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
    pub batch_size: usize,
    pub load_pretrain: bool,
    pub load_parameter: u32,
    pub save_parameter: u32,
    pub grad_stop: bool,
    pub input_size_: usize,
    pub output_size_: usize,
    pub d_one_vec: Option<DeviceBuffer<f32>>
}

impl CUDNNLayer for CUDNNDense {
    fn name_(&mut self) -> &mut String {
        return &mut self.name;
    }

    fn input_desc_(&mut self) -> &mut Option<TensorDescriptor> {
        return &mut self.input_desc;
    }

    fn output_desc_(&mut self) -> &mut Option<TensorDescriptor> {
        return &mut self.output_desc;
    }

    fn filter_desc_(&mut self) -> &mut Option<TensorDescriptor> {
        return &mut self.filter_desc;
    }

    fn bias_desc_(&mut self) -> &mut Option<TensorDescriptor> {
        return &mut self.bias_desc;
    }

    fn input_(&mut self) -> &mut Option<Blob<f32>> {
        return &mut self.input;
    }

    fn output_(&mut self) -> &mut Option<Blob<f32>> {
        return &mut self.output;
    }

    fn grad_input_(&mut self) -> &mut Option<Blob<f32>> {
        return &mut self.grad_input;
    }

    fn grad_output_(&mut self) -> &mut Option<Blob<f32>> {
        return &mut self.grad_output;
    }

    fn freeze_(&mut self) -> &mut bool {
        return &mut self.freeze;
    }

    fn weights_(&mut self) -> &mut Option<Blob<f32>> {
        return &mut self.weights;
    }

    fn biases_(&mut self) -> &mut Option<Blob<f32>> {
        return &mut self.biases;
    }

    fn grad_weights_(&mut self) -> &mut Option<Blob<f32>> {
        return &mut self.grad_weights;
    }

    fn grad_biases_(&mut self) -> &mut Option<Blob<f32>> {
        return &mut self.grad_biases;
    }

    fn gradient_stop_(&mut self) -> &mut bool { return &mut self.grad_stop; }

    fn batch_size_(&mut self) -> &mut usize { return &mut self.batch_size; }

    fn load_pretrain_(&mut self) -> &mut bool { return &mut self.load_pretrain; }

    fn init_weight_bias(&mut self, seed: u32) {
        unsafe { assert_eq!(cudaDeviceSynchronize(), cudaSuccess) }

        if self.weights.is_none() || self.biases.is_none() {
            return;
        }

        let mut gen = match seed {
            0 => { StdRng::from_entropy() }
            _ => { StdRng::seed_from_u64(seed as u64) }
        };

        let range = (6f32 / (self.input.as_ref().unwrap().c * self.input.as_ref().unwrap().h * self.input.as_ref().unwrap().w) as f32).sqrt();
        let distr = Uniform::from(-range..range);

        for i in 0..(self.input.as_ref().unwrap().n * self.input.as_ref().unwrap().c * self.input.as_ref().unwrap().h * self.input.as_ref().unwrap().w) {
            self.weights.as_mut().unwrap().h_ptr[i] = gen.sample(distr);
        }

        for i in 0..(self.input.as_ref().unwrap().n * self.input.as_ref().unwrap().c * self.input.as_ref().unwrap().h * self.input.as_ref().unwrap().w) {
            self.biases.as_mut().unwrap().h_ptr[i] = gen.sample(distr);
        }

        self.weights.as_mut().unwrap().to(cuda);
        self.biases.as_mut().unwrap().to(cuda);

        println!("initialized {} layer", self.name);
    }

    fn update_weights_biases(&mut self, learning_rate: f32) {
        let mut eps = -1f32 * learning_rate;

        if self.weights.is_some() && self.biases.is_some() {
            if DEBUG_UPDATE {
                self.weights.as_mut().unwrap().print(format!("{}: weights", self.name), true, None, None);
                self.grad_weights.as_mut().unwrap().print(format!("{}: gweights", self.name), true, None, None);
            }

            cublas_handle_s.with(|context| {
                let mut weights_len = self.weights.as_mut().unwrap().n * self.weights.as_mut().unwrap().c * self.weights.as_mut().unwrap().h * self.weights.as_mut().unwrap().w;
                rcublas::API::axpy(context.borrow().deref(), &mut eps, self.grad_weights.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(), self.weights.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(), weights_len as i32, Option::from(1), Option::from(1)).expect("Failed to run cublas SAXPY operation");
            });

            if DEBUG_UPDATE {
                self.weights.as_mut().unwrap().print(format!("{}: weights after update", self.name), true, None, None);
            }
        }

        if self.biases.is_some() && self.grad_biases.is_some() {
            if DEBUG_UPDATE {
                self.biases.as_mut().unwrap().print(format!("{}: biases", self.name), true, None, None);
                self.grad_biases.as_mut().unwrap().print(format!("{}: gbiases", self.name), true, None, None);
            }

            cublas_handle_s.with(|context| {
                let mut biases_len = self.biases.as_mut().unwrap().n * self.biases.as_mut().unwrap().c * self.biases.as_mut().unwrap().h * self.biases.as_mut().unwrap().w;
                rcublas::API::axpy(context.borrow().deref(), &mut eps, self.grad_biases.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(), self.biases.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(), biases_len as i32, Option::from(1), Option::from(1)).expect("Failed to run cublas SAXPY operation");
            });

            if DEBUG_UPDATE {
                self.biases.as_mut().unwrap().print(format!("{}: weights after update", self.name), true, None, None);
            }
        }
    }

    fn load_parameter(&mut self) {
        if self.weights.is_some() {
            self.weights.as_mut().unwrap().file_read(format!("{}_weights.banan", self.name.clone())).expect("Failed to read parameter");
        }

        if self.biases.is_some() {
            self.biases.as_mut().unwrap().file_read(format!("{}_biases.banan", self.name.clone())).expect("Failed to read parameter");
        }
    }

    fn save_parameter(&mut self) {
        if self.weights.is_some() {
            self.weights.as_mut().unwrap().file_write(format!("{}_weights.banan", self.name.clone())).expect("Failed to save parameter");
        }

        if self.biases.is_some() {
            self.biases.as_mut().unwrap().file_write(format!("{}_biases.banan", self.name.clone())).expect("Failed to save parameter");
        }
    }

    fn forward(&mut self, input: Box<Blob<f32>>) -> &mut Blob<f32> {
        if self.weights.is_none() {
            self.input_size_ = self.input.as_ref().unwrap().c * self.input.as_ref().unwrap().h * self.input.as_ref().unwrap().w;

            self.weights = Some(Blob::<f32>::new(Some(1), Some(1), Some(self.input_size_), Some(self.output_size_)));
            self.biases = Some(Blob::<f32>::new(Some(1), Some(1), Some(self.input_size_), Some(self.output_size_)));
        }

        todo!()
    }

    fn backward(&mut self, grad_input: Box<Blob<f32>>) -> &mut Blob<f32> {
        todo!()
    }

    fn loss(&mut self, target: Box<Blob<f32>>) -> f32 {
        todo!()
    }

    fn accuracy(&mut self, target: Box<Blob<f32>>) -> u32 {
        todo!()
    }
}