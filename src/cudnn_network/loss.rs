// The following code is stolen and reinterpreted from cudnn mnist sample

use cust::device::DeviceAttribute;
use cust::function::BlockSize;
use cust::memory::{DeviceBox, DeviceMemory};
use cust::prelude::{CopyDestination, DeviceBuffer, Module, Stream, StreamFlags};
use cust::{launch};
use std::mem::size_of;
use lazy_static::lazy_static;

use crate::*;
use crate::cudnn_network;

lazy_static! {
    pub static ref loss_module: Module = Module::from_ptx(String::from(include_str!("../../target/loss.ptx")), &[]).unwrap();
}

pub(crate) struct RegressionLoss {
    pub h_loss: f32,
    pub d_loss: Option<DeviceBox<f32>>,
    pub d_workspace: Option<DeviceBuffer<f32>>,
}

impl RegressionLoss {

    pub(crate) fn new() -> RegressionLoss {
        let loss = 0f32;

        RegressionLoss {
            h_loss: loss,
            d_loss: Some(DeviceBox::new(&loss).expect("Failed to allocate CUDA memory")),
            d_workspace: None,
        }
    }
    pub(crate) fn init_workspace(&mut self, batch_size: usize) {
        if self.d_workspace.is_none() {
            self.d_workspace = Some(DeviceBuffer::zeroed(batch_size).expect("Failed to allocate CUDA memory"));
        }
    }

    pub(crate) fn loss(&mut self, predict: &mut Blob<f32>, target: &mut Blob<f32>) -> f32 {
        let batch_size = target.n;
        let num_outputs = target.c;

        self.init_workspace(batch_size);

        let mse_loss_kernel = loss_module.get_function("mse_loss_kernel").unwrap();
        let device = device_s.with(|device| {device.borrow().unwrap()});

        let num_sms = device.get_attribute(DeviceAttribute::MultiprocessorCount).expect("Failed to get device attribute: MultiprocessorCount") as u32;
        let num_blocks_per_sm = mse_loss_kernel.max_active_blocks_per_multiprocessor(BlockSize {x: cudnn_network::BLOCK_DIM_1D, y: 1, z: 1}, 512 * size_of::<f32>()).expect("Failed to get max active blocks per multiprocessor");

        if cudnn_network::DEBUG_LOSS {
            println!("[[ LOSS ]]");
            predict.print(String::from("predict"), true, Some(batch_size as u32), None);
            target.print(String::from("target"), true, Some(batch_size as u32), None);
        }

        let num_blocks = std::cmp::min(num_blocks_per_sm * num_sms, ((target.c * target.h * target.w) as u32 + cudnn_network::BLOCK_DIM_1D - 1) / cudnn_network::BLOCK_DIM_1D);
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Failed to create multiprocessing stream");

        unsafe {
            launch! (
                mse_loss_kernel<<< num_blocks, cudnn_network::BLOCK_DIM_1D, cudnn_network::BLOCK_DIM_1D * size_of::<f32>() as u32, stream >>>(self.d_loss.as_mut().unwrap().as_raw_ptr(), predict.init_cuda().as_raw_ptr(), target.init_cuda().as_raw_ptr(), self.d_workspace.as_mut().unwrap().as_raw_ptr(), batch_size, num_outputs)
            ).expect("Failed to launch cuda kernel");
        }

        self.d_loss.as_ref().unwrap().copy_to(&mut self.h_loss).expect("Failed to copy loss from cuda device to host");
        return self.h_loss / batch_size as f32;
    }
}