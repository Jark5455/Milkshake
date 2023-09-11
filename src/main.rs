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

    let end = Utc::now().date_naive().and_hms_micro_opt(0, 0, 0, 0).unwrap();
    let start = end - Duration::days(15);

    // let env = StockEnv::new(start, end);

    let mut target = Blob::<f32>::new(Some(5), Some(1), Some(1), Some(1));
    let mut predict = Blob::<f32>::new(Some(5), Some(1), Some(1), Some(1));

    target.h_ptr[0] = 1f32;
    target.h_ptr[1] = 1f32;
    target.h_ptr[2] = 0.2f32;
    target.h_ptr[3] = 0.5f32;
    target.h_ptr[4] = 1f32;

    predict.h_ptr[0] = 0f32;
    predict.h_ptr[1] = 0f32;
    predict.h_ptr[2] = 0f32;
    predict.h_ptr[3] = 0f32;
    predict.h_ptr[4] = 0f32;

    target.init_cuda();
    predict.init_cuda();

    target.to(DeviceType::cuda);
    predict.to(DeviceType::cuda);

    let mut regloss = RegressionLoss::new();
    regloss.init_workspace(5);

    println!("{}", regloss.loss(&mut predict, &mut target))
}
