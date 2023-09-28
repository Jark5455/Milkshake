#![allow(nonstandard_style)]
#![allow(dead_code)]

mod environment;
mod replay_buffer;
mod stockenv;
mod stockframe;
mod td3;
mod tests;

use crate::stockframe::StockFrame;
use dotenv::dotenv;
use lazy_static::lazy_static;
use polars::export::chrono::{Duration, Utc};
use polars::prelude::FillNullStrategy;
use std::sync::Arc;
use tch::nn;
use tch::Device;

lazy_static! {
    static ref device: Arc<Device> = Arc::new(Device::cuda_if_available());
    static ref vs: Arc<nn::VarStore> = Arc::new(nn::VarStore::new(**device));
}

fn main() {
    dotenv().ok();

    let end = Utc::now()
        .date_naive()
        .and_hms_micro_opt(0, 0, 0, 0)
        .unwrap();

    let start = end - Duration::days(15);

    let mut stockframe = StockFrame::new(
        Some(
            vec!["AAPL", "TLSA"]
                .iter()
                .map(|s| String::from(*s))
                .collect(),
        ),
        Some(start.clone()),
        Some(end.clone()),
    );

    stockframe.parse_dt_column();
    stockframe.fill_date_range();
    stockframe.fill_nulls();

    unsafe {
        stockframe.calc_technical_indicators();
    }

    // fill volume, vwap, and trade_count with zeros
    stockframe.frame = Box::new(
        stockframe
            .clone()
            .frame
            .fill_null(FillNullStrategy::Zero)
            .unwrap(),
    );
    stockframe.clean();

    // sort
    stockframe.update_symbol_groups();
    stockframe.frame = Box::new(
        stockframe
            .clone()
            .frame
            .sort(&["symbol", "timestamp"], vec![false, false], false)
            .unwrap(),
    );

    println!("{}", stockframe.frame);
}
