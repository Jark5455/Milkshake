#![allow(nonstandard_style)]
#![allow(dead_code)]

#![macro_use]
extern crate fil_rustacuda;
mod stockframe;
mod environment;
mod stockenv;
mod td3;
mod cudnn_network;

use std::cell::RefCell;
use dotenv::dotenv;
use fil_rustacuda::CudaFlags;
use fil_rustacuda::prelude::{Context, ContextFlags, Device};
use polars::export::chrono::{Duration, Utc};

use crate::stockenv::StockEnv;

fn main() {
    dotenv().ok();
    fil_rustacuda::init(CudaFlags::empty()).expect("Failed to init cuda");
    let device = RefCell::new(Device::get_device(0).expect("Failed to create cuda device"));
    let context = RefCell::new(Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, *device.borrow()).expect("Failed to create cuda context"));

    let end = Utc::now().date_naive().and_hms_micro_opt(0, 0, 0, 0).unwrap();
    let start = end - Duration::days(15);

    let env = StockEnv::new(start, end);
}
