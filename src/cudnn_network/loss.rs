// The following code is stolen and reinterpreted from cudnn mnist sample

use std::ffi::CString;
use fil_rustacuda::memory::{DeviceBox, DeviceBuffer};
use fil_rustacuda::prelude::Module;

struct CrossEntropyLoss {
    h_loss: f32,
    d_loss: Option<DeviceBox<f32>>,
    d_workspace: Option<DeviceBuffer<f32>>
}

impl CrossEntropyLoss {

    pub(crate) fn new() -> CrossEntropyLoss {
        let loss = 0f32;

        CrossEntropyLoss {
            h_loss: loss,
            d_loss: Some(DeviceBox::new(&loss).expect("Failed to allocate CUDA memory")),
            d_workspace: None
        }
    }
    pub(crate) fn init_workspace(&mut self, batch_size: u32) {
        if self.d_workspace.is_none() {
            unsafe {
                self.d_workspace = Some(DeviceBuffer::uninitialized(batch_size as usize).expect("Failed to allocate CUDA memory"));
            }
        }
    }

    pub(crate) fn load_kernel(&mut self) {
        let c_path = CString::new(include_str!("../../target/loss.ptx")).unwrap();
        let module = Module::load_from_string(&c_path).unwrap();
    }
}