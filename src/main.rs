#![allow(nonstandard_style)]
#![allow(dead_code)]
mod stockframe;
mod environment;
mod stockenv;

use dotenv::dotenv;
use polars::export::chrono::{Duration, Utc};
use stockenv::StockEnv;
use std::ops::Sub;

fn main() {
    dotenv().ok();

    let end = Utc::now().date_naive().and_hms_micro_opt(0, 0, 0, 0).unwrap();
    let start = end.sub(Duration::days(15));

    let env = StockEnv::new(start, end);
    println!("{:?}", env.stockframe.frame);
}
