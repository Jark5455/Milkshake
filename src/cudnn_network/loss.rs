// The following code is stolen and reinterpreted from cudnn mnist sample

use cust::device::DeviceAttribute;
use cust::function::BlockSize;
use cust::memory::DeviceBox;
use cust::prelude::{DeviceBuffer, Module};
use std::ffi::CString;
use std::mem::size_of;
use std::borrow::Borrow;
use std::ops::Deref;

use crate::cudnn_network::{blob, BLOCK_DIM_1D};
use crate::{device_s};

struct CrossEntropyLoss {
    h_loss: f32,
    d_loss: Option<DeviceBox<f32>>,
    d_workspace: Option<DeviceBuffer<f32>>,
    kernel_module: Module
}

impl CrossEntropyLoss {

    pub(crate) fn new() -> CrossEntropyLoss {
        let loss = 0f32;

        let ptxcode = CString::new(include_str!("../../target/loss.ptx")).unwrap();
        let module = Module::load_from_string(&ptxcode).unwrap();

        CrossEntropyLoss {
            h_loss: loss,
            d_loss: Some(DeviceBox::new(&loss).expect("Failed to allocate CUDA memory")),
            d_workspace: None,
            kernel_module: module
        }
    }
    pub(crate) fn init_workspace(&mut self, batch_size: u32) {
        if self.d_workspace.is_none() {
            unsafe {
                self.d_workspace = Some(DeviceBuffer::uninitialized(batch_size as usize).expect("Failed to allocate CUDA memory"));
            }
        }
    }

    pub(crate) fn loss(&mut self, predict: &blob::Blob<f32>, target: &blob::Blob<f32>) {
        let device = device_s.with(|device| {*device.borrow()});

        let num_sms = device.get_attribute(DeviceAttribute::MultiprocessorCount).expect("Failed to get device attribute: MultiprocessorCount");
        let num_blocks_per_sm = self.kernel_module.get_function("softmax_loss_kernel").unwrap().max_active_blocks_per_multiprocessor(BlockSize {x: BLOCK_DIM_1D, y: 1, z: 1}, 512 * size_of::<f32>());
    }
}