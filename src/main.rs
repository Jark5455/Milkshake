#![allow(nonstandard_style)]
#![allow(dead_code)]

#![macro_use]
extern crate fil_rustacuda;
mod stockframe;
mod environment;
mod stockenv;
mod td3;
mod cudnn_network;

use dotenv::dotenv;
use polars::export::chrono::{Duration, Utc};

use crate::stockenv::StockEnv;

fn main() {
    dotenv().ok();

    let end = Utc::now().date_naive().and_hms_micro_opt(0, 0, 0, 0).unwrap();
    let start = end - Duration::days(15);

    let env = StockEnv::new(start, end);
}
