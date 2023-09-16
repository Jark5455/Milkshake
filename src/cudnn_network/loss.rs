// The following code is stolen and reinterpreted from cudnn mnist sample

use std::ffi::c_int;
use std::mem::size_of;
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::driver::result::occupancy::max_active_block_per_multiprocessor;
use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;

use crate::*;
use crate::cudnn_network;

pub(crate) struct RegressionLoss {
    pub host_loss: f32,
    pub device_loss: CudaSlice<f32>,
    pub device_workspace: Option<CudaSlice<f32>>,
}

impl RegressionLoss {

    pub(crate) fn new() -> RegressionLoss {
        let loss = 0f32;

        RegressionLoss {
            host_loss: loss,
            device_loss: device.alloc_zeros(1).expect("Failed to allocate device loss value"),
            device_workspace: None,
        }
    }
    pub(crate) fn init_workspace(&mut self, batch_size: usize) {
        if self.device_workspace.is_none() {
            self.device_workspace = Some(device.alloc_zeros(batch_size).expect("Failed to allocate device loss workspace"));
        }
    }

    pub(crate) fn loss(&mut self, predict: &mut Blob<f32>, target: &mut Blob<f32>) -> f32 {
        let batch_size = target.n;
        let num_outputs = target.c;

        self.init_workspace(batch_size);


        let mse_loss_kernel = device.get_func("mse_loss", "mse_loss_kernel").expect("Could not find mse loss kernel function");

        let num_sms = device.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).expect("Failed to query CUDA device multiprocessor count");
        let num_blocks_per_sm = mse_loss_kernel.occupancy_max_active_blocks_per_multiprocessor(cudnn_network::BLOCK_DIM_1D, (cudnn_network::BLOCK_DIM_1D * size_of::<f32>()) as usize, None).expect("Failed to query max active blocks per multiprocessor");

        if cudnn_network::DEBUG_LOSS {
            println!("[[ LOSS ]]");
            predict.print(String::from("predict"), true, Some(batch_size as u32), None);
            target.print(String::from("target"), true, Some(batch_size as u32), None);
        }

        let num_blocks = std::cmp::min(num_blocks_per_sm * num_sms, ((target.c * target.h * target.w) as u32 + cudnn_network::BLOCK_DIM_1D - 1) / cudnn_network::BLOCK_DIM_1D);

        let launch_config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (cudnn_network::BLOCK_DIM_1D, 1, 1),
            shared_mem_bytes: cudnn_network::BLOCK_DIM_1D * size_of::<f32>(),
        };

        unsafe {
            mse_loss_kernel.launch(launch_config, (
                &mut self.device_loss, predict.cuda(), target.cuda(), &mut self.device_workspace.as_mut().unwrap(), batch_size, num_outputs)
            ).expect("Failed to launch cuda loss kernel");
        }

        // maybe sync here? idk

        self.host_loss = device.dtoh_sync_copy(&self.device_loss).expect("Failed to copy loss from device to host")[0];
        return self.host_loss / batch_size as f32;
    }
}