// The following code is stolen and reinterpreted from cudnn mnist sample

use anyhow::Result;
use std::ops::Deref;
use cust::launch;
use cust::memory::{DeviceBuffer, DeviceMemory, GpuBuffer};
use cust::prelude::{Stream, StreamFlags, Module};
use lazy_static::lazy_static;
use rand::distributions::Uniform;
use rand::prelude::{StdRng};
use rand::{Rng, SeedableRng};
use rcublas::api::Operation;
use rcudnn::{cudaDeviceSynchronize, TensorDescriptor};
use rcudnn::cudaError_t::cudaSuccess;
use crate::{context_s, cublas_handle_s};
use crate::cudnn_network::blob::Blob;
use crate::cudnn_network::blob::DeviceType::cuda;
use crate::cudnn_network::{DEBUG_DENSE, DEBUG_UPDATE, one, zero};
use crate::cudnn_network::BLOCK_DIM_1D;

lazy_static! {
    pub static ref cudnn_network_module: Module = Module::from_ptx(String::from(include_str!("../../target/cudnn_network.ptx")), &[]).unwrap();
}

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
    fn batch_size_(&mut self) -> &mut usize;
    fn load_pretrain_(&mut self) -> &mut bool;

    fn init_weight_bias(&mut self, seed: u32);
    fn update_weights_biases(&mut self, learning_rate: f32);
    fn load_parameter(&mut self) -> Result<()>;
    fn save_parameter(&mut self) -> Result<()>;

    fn forward(&mut self, input: Blob<f32>) -> &mut Blob<f32>;
    fn backward(&mut self, grad_output: Blob<f32>) -> &mut Blob<f32>;
    fn loss(&mut self, target: Blob<f32>) -> f32;
    fn accuracy(&mut self, target: Blob<f32>) -> u32;
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
    pub load_pretrain: bool,
    pub batch_size: usize,
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

    fn load_parameter(&mut self) -> Result<()> {
        if self.weights.is_some() {
            self.weights.as_mut().unwrap().file_read(format!("{}_weights.banan", self.name.clone()))?;
        }

        if self.biases.is_some() {
            self.biases.as_mut().unwrap().file_read(format!("{}_biases.banan", self.name.clone()))?;
        }

        Ok(())
    }

    fn save_parameter(&mut self) -> Result<()> {
        if self.weights.is_some() {
            self.weights.as_mut().unwrap().file_write(format!("{}_weights.banan", self.name.clone()))?;
        }

        if self.biases.is_some() {
            self.biases.as_mut().unwrap().file_write(format!("{}_biases.banan", self.name.clone()))?;
        }

        Ok(())
    }

    fn forward(&mut self, input: Blob<f32>) -> &mut Blob<f32> {
        if self.weights.is_none() {
            self.input_size_ = self.input.as_ref().unwrap().c * self.input.as_ref().unwrap().h * self.input.as_ref().unwrap().w;

            self.weights = Some(Blob::<f32>::new(Some(1), Some(1), Some(self.input_size_), Some(self.output_size_)));
            self.biases = Some(Blob::<f32>::new(Some(1), Some(1), Some(self.input_size_), Some(self.output_size_)));
        }

        if self.input.is_none() || self.batch_size != input.n {
            self.input = Some(input);
            self.batch_size = self.input.as_ref().unwrap().n;

            if self.output.is_none() {
                self.output = Some(Blob::<f32>::new(Some(self.batch_size), Some(self.output_size_), None, None));
            } else {
                self.output.as_mut().unwrap().reset(Some(self.batch_size), Some(self.output_size_), None, None);
            }

            self.output.as_mut().unwrap().init_tensor();
            self.d_one_vec = Some(DeviceBuffer::zeroed(self.batch_size).unwrap());

            let init_one_vec_kernel = cudnn_network_module.get_function("init_one_vec").unwrap();
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create multiprocessing stream");

            unsafe {
                launch!(
                    init_one_vec_kernel<<<(self.batch_size as u32 + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D, 0, stream>>>(self.d_one_vec.as_mut().unwrap().as_raw_ptr(), self.batch_size)
                ).expect("Failed to launch init_one_vec kernel");
            }

            if self.load_pretrain && !self.freeze {
                self.load_parameter().expect("Error occured while loading pretrain");
            } else if !self.freeze {
                self.init_weight_bias(0);
            }
        }

        cublas_handle_s.with(|cublas| {
            rcublas::API::gemm(
                cublas.borrow().deref(),
                Operation::Trans,
                Operation::NoTrans,
                self.output_size_ as i32,
                self.batch_size as i32,
                self.input_size_ as i32,
                &mut one,
                self.weights.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                self.input_size_ as i32,
                self.input.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                self.input_size_ as i32,
                &mut zero,
                self.output.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                self.output_size_ as i32
            ).expect("Failed to run cublas SGEMM (weights^T * input)");

            rcublas::API::gemm(
                cublas.borrow().deref(),
                Operation::NoTrans,
                Operation::NoTrans,
                self.output_size_ as i32,
                self.batch_size as i32,
                1,
                &mut one,
                self.biases.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                self.output_size_ as i32,
                self.d_one_vec.as_mut().unwrap().as_device_ptr().as_mut_ptr(),
                1,
                &mut one,
                self.output.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                self.output_size_ as i32
            ).expect("Failed to run cublas SGEMM (biases * d_one_vec^T)");
        });

        if DEBUG_DENSE {
            self.input.as_mut().unwrap().print(String::from("input"), true, None, None);
            self.weights.as_mut().unwrap().print(String::from("weights"), true, None, None);
            self.biases.as_mut().unwrap().print(String::from("biases"), true, None, None);
            self.output.as_mut().unwrap().print(String::from("output"), true, None, None);
        }

        return self.output.as_mut().unwrap();
    }

    fn backward(&mut self, grad_output: Blob<f32>) -> &mut Blob<f32> {
        if self.grad_weights.is_none() {
            self.weights = Some(Blob::<f32>::new(Some(self.weights.as_mut().unwrap().n), Some(self.weights.as_mut().unwrap().c), Some(self.weights.as_mut().unwrap().h), Some(self.weights.as_mut().unwrap().w)));
            self.biases = Some(Blob::<f32>::new(Some(self.biases.as_mut().unwrap().n), Some(self.biases.as_mut().unwrap().c), Some(self.biases.as_mut().unwrap().h), Some(self.biases.as_mut().unwrap().w)));
        }

        if self.grad_input.is_none() || self.batch_size != self.grad_output.as_mut().unwrap().n {
            self.grad_output = Some(grad_output);

            if self.grad_input.is_none() {
                self.grad_input = Some(Blob::<f32>::new(Some(self.input.as_mut().unwrap().n), Some(self.input.as_mut().unwrap().c), Some(self.input.as_mut().unwrap().h), Some(self.input.as_mut().unwrap().w)));
            } else {
                self.grad_input.as_mut().unwrap().reset(Some(self.input.as_mut().unwrap().n), Some(self.input.as_mut().unwrap().c), Some(self.input.as_mut().unwrap().h), Some(self.input.as_mut().unwrap().w));
            }
        }

        cublas_handle_s.with(|cublas| {
            // we dont have sgemmv so we treat the vector as a 1 col matrix, hope this works ig
            rcublas::API::gemm(
                cublas.borrow().deref(),
                Operation::NoTrans,
                Operation::NoTrans,
                self.output_size_ as i32,
                1,
                self.batch_size as i32,
                &mut one,
                self.grad_output.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                self.output_size_ as i32,
                self.d_one_vec.as_mut().unwrap().as_device_ptr().as_mut_ptr(),
                1,
                &mut zero,
                self.grad_biases.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                1
            ).expect("Failed to run cublas SGEMMV (db = (dy) * d_one_vec)");

            rcublas::API::gemm(
                cublas.borrow().deref(),
                Operation::NoTrans,
                Operation::Trans,
                self.input_size_ as i32,
                self.output_size_ as i32,
                self.batch_size as i32,
                &mut one,
                self.input.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                self.input_size_ as i32,
                self.grad_output.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                self.output_size_ as i32,
                &mut zero,
                self.grad_weights.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                self.input_size_ as i32
            ).expect("Failed to run cublas SGEMM (dw = x * (dy)^T)");

            if self.grad_stop {
                rcublas::API::gemm(
                    cublas.borrow().deref(),
                    Operation::NoTrans,
                    Operation::NoTrans,
                    self.input_size_ as i32,
                    self.batch_size as i32,
                    self.output_size_ as i32,
                    &mut one,
                    self.weights.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                    self.input_size_ as i32,
                    self.grad_output.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                    self.output_size_ as i32,
                    &mut zero,
                    self.grad_input.as_mut().unwrap().init_cuda().as_device_ptr().as_mut_ptr(),
                    self.input_size_ as i32
                ).expect("Failed to run cublas SGEMM (dx = W * dy)");
            }
        });

        return self.grad_input.as_mut().unwrap();
    }

    fn loss(&mut self, target: Blob<f32>) -> f32 {
        todo!()
    }

    fn accuracy(&mut self, target: Blob<f32>) -> u32 {
        todo!()
    }
}