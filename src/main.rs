#![allow(nonstandard_style)]
#![allow(dead_code)]

mod stockframe;
mod environment;
mod stockenv;
mod td3;
mod cudnn_network;
mod tests;

use cust::prelude::{Device, Context};
use cust::CudaFlags;
use dotenv::dotenv;
use polars::export::chrono::{Duration, Utc};
use std::cell::RefCell;
use std::ptr;

use crate::cudnn_network::blob::{Blob, DeviceType};
use crate::cudnn_network::loss::RegressionLoss;

use crate::stockenv::StockEnv;

thread_local! {
    pub static device_s: RefCell<Option<Device>> = RefCell::new(None);
    pub static context_s: RefCell<Option<Context>> = RefCell::new(None);
    pub static cudnn_handle_s: RefCell<rcudnn::Cudnn> = RefCell::new(rcudnn::Cudnn::new().expect("Failed to create cudnn handle"));
    pub static cublas_handle_s: RefCell<rcublas::Context> = RefCell::new(rcublas::Context::new().expect("Failed to create cublas handle"));
}

fn main() {
    dotenv().ok();
    cust::init(CudaFlags::empty()).expect("Failed to initialize cuda");

    device_s.with(|device_ref| {
        device_ref.replace(Some(Device::get_device(0).expect("Failed to find cuda device")));

        context_s.with(|context_ref| {
            context_ref.replace(Some(Context::new(device_ref.borrow().unwrap()).expect("Failed to create cuda context")))
        });
    });


}
