#![allow(nonstandard_style)]
#![allow(dead_code)]

mod stockframe;
mod environment;
mod stockenv;
mod td3;
mod cudnn_network;

use cust::prelude::{Device, Context};
use cust::CudaFlags;
use dotenv::dotenv;
use polars::export::chrono::{Duration, Utc};
use std::cell::RefCell;
use std::ops::Deref;

use crate::stockenv::StockEnv;

thread_local! {
    pub static device_s: RefCell<Device> = RefCell::new(Device::get_device(0).expect("Failed to find cuda device"));
    pub static context_s: RefCell<Context> = RefCell::new(Context::new(device_s.with(|device| {*device.borrow()})).expect("Failed to create cuda context"));
    pub static cudnn_handle_s: RefCell<rcudnn::Cudnn> = RefCell::new(rcudnn::Cudnn::new().expect("Failed to create cudnn handle"));
    pub static cublas_handle_s: RefCell<rcublas::Context> = RefCell::new(rcublas::Context::new().expect("Failed to create cublas handle"));
}

fn main() {
    dotenv().ok();
    cust::init(CudaFlags::empty()).expect("Failed to initialize cuda");

    let end = Utc::now().date_naive().and_hms_micro_opt(0, 0, 0, 0).unwrap();
    let start = end - Duration::days(15);

    let env = StockEnv::new(start, end);
}
