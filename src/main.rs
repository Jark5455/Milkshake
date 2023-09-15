#![allow(nonstandard_style)]
#![allow(dead_code)]

mod stockframe;
mod environment;
mod stockenv;
mod td3;
mod cudnn_network;
mod tests;

use dotenv::dotenv;

use std::sync::Arc;
use cudarc::cublas::safe::CudaBlas;
use cudarc::driver::safe::CudaDevice;
use cudarc::cudnn::safe::Cudnn;
use cudarc::nvrtc::compile_ptx;
use lazy_static::lazy_static;

use crate::cudnn_network::blob::{Blob};
use crate::cudnn_network::blob::DeviceType::cuda;
use crate::cudnn_network::loss::RegressionLoss;

lazy_static! {
    static ref device: Arc<CudaDevice> = CudaDevice::new(0).unwrap();
    static ref cublas: Arc<CudaBlas> = CudaBlas::new(device);
    static ref cudnn: Arc<Cudnn> = Cudnn::new(device);

    static ref mse_loss_kernel_code: String = String::from("\n
        extern \"C\" {
            __global__ void
            mse_loss_kernel(float *reduced_loss, float *predict, float *target, float *workspace, int batch_size, int num_outputs)
            {
                int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

                extern __shared__ float s_data[];
                float loss = 0.f;

                // each thread squared error for each data and accumulate to shared memory
                for (int c = 0; c < num_outputs; c++)
                    loss += pow(target[batch_idx * num_outputs + c] - predict[batch_idx * num_outputs + c], 2);
                workspace[batch_idx] = loss;

                // then, we do reduction the result to calculate loss using 1 thread block
                if (blockIdx.x > 0) return;

                // cumulate workspace data
                s_data[threadIdx.x] = 0.f;
                for (int i = 0; i < batch_size; i += blockDim.x)
                {
                    s_data[threadIdx.x] += workspace[threadIdx.x + i];
                }

                __syncthreads();

                // reduction
                for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
                {
                    if (threadIdx.x + stride < batch_size)
                        s_data[threadIdx.x] += s_data[threadIdx.x + stride];

                    __syncthreads();
                }

                if (threadIdx.x == 0) {
                    reduced_loss[blockIdx.x] = s_data[0];
                }
            }
        }
    ");

    static ref init_one_vec_kernel_code: String = String::from("\n
        __global__ void init_one_vec(float* d_one_vec, size_t length)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= length) return;

            d_one_vec[i] = 1.f;
        }
    ");
}

fn main() {
    dotenv().ok();

    let mse_loss_kernel = compile_ptx(mse_loss_kernel_code.clone()).expect("Failed to compile mse loss kernel");
    device.load_ptx(mse_loss_kernel, "mse_loss", &["mse_loss_kernel"]).expect("Failed to load mse loss kernel");

    let init_one_vec_kernel = compile_ptx(init_one_vec_kernel_code.clone()).expect("Failed to compile init one vec kernel");
    device.load_ptx(init_one_vec_kernel, "init_one_vec", &["init_one_vec"]).expect("Failed to load init one vec kernel");

    /*

    cust::init(CudaFlags::empty()).expect("Failed to initialize cuda");

    device_s.with(|device_ref| {
        device_ref.replace(Some(Device::get_device(0).expect("Failed to find cuda device")));

        context_s.with(|context_ref| {
            context_ref.replace(Some(Context::new(device_ref.borrow().unwrap()).expect("Failed to create cuda context")))
        });
    });

    */
}
