#![allow(nonstandard_style)]
#![allow(dead_code)]
mod stockframe;
mod environment;
mod stockenv;

use dotenv::dotenv;
use polars::export::chrono::{Duration, Utc};

use crate::stockenv::StockEnv;
use crate::environment::Environment;

fn main() {
    dotenv().ok();

    let end = Utc::now().date_naive().and_hms_micro_opt(0, 0, 0, 0).unwrap();
    let start = end - Duration::days(15);

    let mut env = StockEnv::new(start, end);
}
