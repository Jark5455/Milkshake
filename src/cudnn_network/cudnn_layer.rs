// The following code is stolen and reinterpreted from cudnn mnist sample

use std::ffi::c_int;
use anyhow::Result;
use cudarc::cublas::{Axpy, AxpyConfig, Gemm, GemmConfig, Gemv, GemvConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cudnn::TensorDescriptor;
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
use rand::distributions::Uniform;
use rand::prelude::{StdRng};
use rand::{Rng, SeedableRng};
use crate::cudnn_network::blob::Blob;
use crate::cudnn_network::blob::DeviceType::cuda;
use crate::cudnn_network::{BLOCK_DIM_1D, DEBUG_BACKWARD, DEBUG_DENSE, DEBUG_UPDATE};
use crate::{cublas, device};

trait CUDNNLayer {
    fn name_(&mut self) -> &mut String;
    fn input_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>>;
    fn output_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>>;
    fn filter_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>>;
    fn bias_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>>;

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

    fn init_weight_bias(&mut self, seed: u32) {
        device.synchronize().expect("Failed to sync device, previous stuff prolly failed");

        if self.weights_().is_none() || self.biases_().is_none() {
            return;
        }

        let mut gen = match seed {
            0 => { StdRng::from_entropy() }
            _ => { StdRng::seed_from_u64(seed as u64) }
        };

        let range = (6f32 / (self.input_().as_ref().unwrap().c * self.input_().as_ref().unwrap().h * self.input_().as_ref().unwrap().w) as f32).sqrt();
        let distr = Uniform::from(-range..range);

        for i in 0..(self.input_().as_ref().unwrap().n * self.input_().as_ref().unwrap().c * self.input_().as_ref().unwrap().h * self.input_().as_ref().unwrap().w) {
            self.weights_().as_mut().unwrap().host_slice[i] = gen.sample(distr);
        }

        for i in 0..(self.input_().as_ref().unwrap().n * self.input_().as_ref().unwrap().c * self.input_().as_ref().unwrap().h * self.input_().as_ref().unwrap().w) {
            self.biases_().as_mut().unwrap().host_slice[i] = gen.sample(distr);
        }

        self.weights_().as_mut().unwrap().to(cuda);
        self.biases_().as_mut().unwrap().to(cuda);

        println!("initialized {} layer", self.name_());
    }
    fn update_weights_biases(&mut self, learning_rate: f32) {
        let mut eps = -1f32 * learning_rate;
        let name = self.name_().clone();

        if self.weights_().is_some() && self.biases_().is_some() {
            if DEBUG_UPDATE {
                self.weights_().as_mut().unwrap().print(format!("{}: weights", name), true, None, None);
                self.grad_weights_().as_mut().unwrap().print(format!("{}: gweights", name), true, None, None);
            }

            let cfg = AxpyConfig {
                alpha: eps,
                n: self.weights_().as_mut().unwrap().len() as std::ffi::c_int,
                incx: 1,
                incy: 1,
            };

            unsafe {
                let mut grad_weights = self.grad_weights_().as_mut().unwrap().cuda().clone();
                cublas.axpy(cfg, &mut grad_weights, self.weights_().as_mut().unwrap().cuda()).expect("Failed to run cuda AXPY");
            }

            if DEBUG_UPDATE {
                self.weights_().as_mut().unwrap().print(format!("{}: weights after update", name), true, None, None);
            }
        }
    }

    fn load_parameter(&mut self) -> Result<()> {
        let name = self.name_().clone();

        if self.weights_().is_some() {
            self.weights_().as_mut().unwrap().file_read(format!("{}_weights.banan", name))?;
        }

        if self.biases_().is_some() {
            self.biases_().as_mut().unwrap().file_read(format!("{}_biases.banan", name))?;
        }

        Ok(())
    }

    fn save_parameter(&mut self) -> Result<()> {
        let name = self.name_().clone();

        if self.weights_().is_some() {
            self.weights_().as_mut().unwrap().file_write(format!("{}_weights.banan", name))?;
        }

        if self.biases_().is_some() {
            self.biases_().as_mut().unwrap().file_write(format!("{}_biases.banan", name))?;
        }

        Ok(())
    }

    fn loss(&mut self, target: Blob<f32>) -> f32 {
        panic!("this layer ({}) doesnt have loss, you did something wrong", self.name_().clone());
    }

    fn accuracy(&mut self, target: Blob<f32>) -> u32 {
        panic!("this layer ({}) doesnt have loss, you cant estimate accuracy without loss", self.name_().clone());
    }

    fn forward(&mut self, input: Blob<f32>) -> &mut Blob<f32>;
    fn backward(&mut self, grad_output: Blob<f32>) -> &mut Blob<f32>;
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
    pub input_desc: Option<TensorDescriptor<f32>>,
    pub output_desc: Option<TensorDescriptor<f32>>,
    pub filter_desc: Option<TensorDescriptor<f32>>,
    pub bias_desc: Option<TensorDescriptor<f32>>,
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
    pub d_one_vec: Option<CudaSlice<f32>>
}

/*
struct CUDNNActivation {
    pub name: String,
    pub input_desc: Option<TensorDescriptor<f32>>,
    pub output_desc: Option<TensorDescriptor<f32>>,
    pub filter_desc: Option<TensorDescriptor<f32>>,
    pub bias_desc: Option<TensorDescriptor<f32>>,
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
    pub coef: f32,
    pub act_desc: ,
    pub mode: cudnnActivationMode_t
}
*/

impl CUDNNDense {
    pub(crate) fn new(name: String, output_size: usize) -> CUDNNDense {
        CUDNNDense {
            name: name,
            input_desc: None,
            output_desc: None,
            filter_desc: None,
            bias_desc: None,
            input: None,
            output: None,
            grad_input: None,
            grad_output: None,
            freeze: false,
            weights: None,
            biases: None,
            grad_weights: None,
            grad_biases: None,
            load_pretrain: false,
            batch_size: 0,
            grad_stop: false,
            input_size_: 0,
            output_size_: output_size,
            d_one_vec: None,
        }
    }
}

impl CUDNNLayer for CUDNNDense {
    fn name_(&mut self) -> &mut String {
        return &mut self.name;
    }

    fn input_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>> {
        return &mut self.input_desc;
    }

    fn output_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>> {
        return &mut self.output_desc;
    }

    fn filter_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>> {
        return &mut self.filter_desc;
    }

    fn bias_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>> {
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

    fn gradient_stop_(&mut self) -> &mut bool {
        return &mut self.grad_stop;
    }

    fn batch_size_(&mut self) -> &mut usize {
        return &mut self.batch_size;
    }

    fn load_pretrain_(&mut self) -> &mut bool {
        return &mut self.load_pretrain;
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

            self.output.as_mut().unwrap().tensor();
            self.d_one_vec = Some(device.alloc_zeros(self.batch_size).expect("Failed to allocate d_one_vec"));

            let init_one_vec_kernel = device.get_func("init_one_vec", "init_one_vec").expect("Could not find init_one_vec function");

            let cfg = LaunchConfig {
                grid_dim: ((self.batch_size as u32 + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, 1, 1),
                block_dim: (BLOCK_DIM_1D, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                init_one_vec_kernel.launch(cfg, (
                    self.d_one_vec.as_mut().unwrap(), self.batch_size as u32)
                ).expect("Failed to launch cuda loss kernel");
            }

            if self.load_pretrain && !self.freeze {
                self.load_parameter().expect("Error occured while loading pretrain");
            } else if !self.freeze {
                self.init_weight_bias(0);
            }
        }

        let gemmConfig = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.output_size_ as c_int,
            n: self.batch_size as c_int,
            k: self.input_size_ as c_int,
            alpha: 1f32,
            lda: self.input_size_ as c_int,
            ldb: self.input_size_ as c_int,
            beta: 0f32,
            ldc: self.output_size_ as c_int,
        };

        unsafe { cublas.gemm(gemmConfig, self.weights.as_mut().unwrap().cuda(), self.input.as_mut().unwrap().cuda(), self.output.as_mut().unwrap().cuda()) }.expect("Failed to run cublas SGEMM (weights^T * input)");

        let gemmConfig = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.output_size_ as c_int,
            n: self.batch_size as c_int,
            k: 1,
            alpha: 1f32,
            lda: self.output_size_ as c_int,
            ldb: 1,
            beta: 0f32,
            ldc: self.output_size_ as c_int,
        };

        unsafe { cublas.gemm(gemmConfig, self.biases.as_mut().unwrap().cuda(), self.d_one_vec.as_mut().unwrap(), self.output.as_mut().unwrap().cuda()) }.expect("Failed to run cublas SGEMM (output += biases * d_one_vec^T)");

        if DEBUG_DENSE {
            self.input.as_mut().unwrap().print(format!("{}::input", self.name), true, None, None);
            self.weights.as_mut().unwrap().print(format!("{}::weight", self.name), true, None, None);
            self.biases.as_mut().unwrap().print(format!("{}::bias", self.name), true, None, None);
            self.output.as_mut().unwrap().print(format!("{}::output", self.name), true, None, None);
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

        let gemvConfig = GemvConfig {
            trans: cublasOperation_t::CUBLAS_OP_N,
            m: self.output_size_ as c_int,
            n: self.batch_size as c_int,
            alpha: 1f32,
            lda: self.output_size_ as c_int,
            incx: 1,
            beta: 0f32,
            incy: 1,
        };

        unsafe {
            cublas.gemv(
                gemvConfig,
                self.grad_output.as_mut().unwrap().cuda(),
                self.d_one_vec.as_mut().unwrap(),
                self.grad_biases.as_mut().unwrap().cuda()
            )
        }.expect("Failed to run cublas SGEMMV (db = (dy) * d_one_vec)");

        let gemmConfig = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: self.input_size_ as c_int,
            n: self.output_size_ as c_int,
            k: self.batch_size as c_int,
            alpha: 1f32,
            lda: self.input_size_ as c_int,
            ldb: self.output_size_ as c_int,
            beta: 0f32,
            ldc: self.input_size_ as c_int,
        };

        unsafe {
            cublas.gemm(
                gemmConfig,
                self.input.as_mut().unwrap().cuda(),
                self.grad_output.as_mut().unwrap().cuda(),
                self.grad_weights.as_mut().unwrap().cuda()
            )
        }.expect("Failed to run cublas SGEMM (dw = x * (dy)^T)");

        if self.grad_stop {
            let gemmConfig = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: self.input_size_ as c_int,
                n: self.batch_size as c_int,
                k: self.output_size_ as c_int,
                alpha: 1f32,
                lda: self.input_size_ as c_int,
                ldb: self.output_size_ as c_int,
                beta: 0f32,
                ldc: self.input_size_ as c_int,
            };

            unsafe {
                cublas.gemm(
                    gemmConfig,
                    self.weights.as_mut().unwrap().cuda(),
                    self.grad_output.as_mut().unwrap().cuda(),
                    self.grad_input.as_mut().unwrap().cuda()
                )
            }.expect("Failed to run cublas SGEMM (dx = W * dy)");
        }

        if DEBUG_BACKWARD {
            println!("[BACKWARD]");
            self.grad_output.as_mut().unwrap().print(format!("{}::gradients", self.name), true, None, None);
            self.weights.as_mut().unwrap().print(format!("{}::gfilter", self.name), true, None, None);
            self.biases.as_mut().unwrap().print(format!("{}::gbias", self.name), true, None, None);

            if !self.grad_stop {
                self.grad_input.as_mut().unwrap().print(format!("{}::gdata", self.name), true, None, None);
            }
        }

        return self.grad_input.as_mut().unwrap();
    }
}

/*
impl CUDNNActivation {
    pub(crate) fn new(name: String, activation: cudnnActivationMode_t, coef: Option<f32>) -> CUDNNActivation {
        let coefficient = coef.unwrap_or(0f32);
        let activation_desc = ActivationDescriptor::new(activation).expect("Failed to create activation descriptor");

        return CUDNNActivation {
            name: name,
            input_desc: None,
            output_desc: None,
            filter_desc: None,
            bias_desc: None,
            input: None,
            output: None,
            grad_input: None,
            grad_output: None,
            freeze: false,
            weights: None,
            biases: None,
            grad_weights: None,
            grad_biases: None,
            load_pretrain: false,
            batch_size: 0,
            grad_stop: false,
            coef: coefficient,
            act_desc: activation_desc,
            mode: activation,
        }
    }
}

impl CUDNNLayer for CUDNNActivation {
    fn name_(&mut self) -> &mut String {
        return &mut self.name;
    }

    fn input_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>> {
        return &mut self.input_desc;
    }

    fn output_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>> {
        return &mut self.output_desc;
    }

    fn filter_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>> {
        return &mut self.filter_desc;
    }

    fn bias_desc_(&mut self) -> &mut Option<TensorDescriptor<f32>> {
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

    fn gradient_stop_(&mut self) -> &mut bool {
        return &mut self.grad_stop;
    }

    fn batch_size_(&mut self) -> &mut usize {
        return &mut self.batch_size;
    }

    fn load_pretrain_(&mut self) -> &mut bool {
        return &mut self.load_pretrain;
    }

    fn forward(&mut self, input: Blob<f32>) -> &mut Blob<f32> {
        if self.input.is_none() || self.batch_size != self.input.as_mut().unwrap().n {
            self.input = Some(input);
            self.input_desc = Some(self.input.as_mut().unwrap().init_tensor().clone());
            self.batch_size = self.input.as_mut().unwrap().n;

            if self.output.is_none() {
                self.output = Some(Blob::<f32>::new(Some(self.input.as_mut().unwrap().n), Some(self.input.as_mut().unwrap().c), Some(self.input.as_mut().unwrap().h), Some(self.input.as_mut().unwrap().w)));
            } else {
                self.output.as_mut().unwrap().reset(Some(self.input.as_mut().unwrap().n), Some(self.input.as_mut().unwrap().c), Some(self.input.as_mut().unwrap().h), Some(self.input.as_mut().unwrap().w));
            }

            cudnn_handle_s.with(|cudnn| {
                API::activation_forward(*cudnn.borrow().id_c(),
                                        *self.act_desc.id_c(),
                                        &one as *const f32 as *const c_void,
                                        *self.input_desc.as_mut().unwrap().id_c(),
                                        self.input.as_mut().unwrap().init_cuda().as_raw_ptr() as *const c_void,
                                        &zero as *const f32 as *const c_void,
                                        *self.output_desc.as_mut().unwrap().id_c(),
                                        self.output.as_mut().unwrap().init_cuda().as_raw_ptr() as *mut c_void
                ).expect("Failed to run activation method");
            });
        }

        return self.output.as_mut().unwrap();
    }

    fn backward(&mut self, grad_output: Blob<f32>) -> &mut Blob<f32> {
        return self.grad_input.as_mut().unwrap();
    }
}
*/