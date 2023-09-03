#![allow(nonstandard_style)]
#![allow(dead_code)]
mod stockframe;
mod environment;
mod stockenv;

use dotenv::dotenv;
use polars::prelude::{FillNullStrategy};
use stockframe::StockFrame;

fn main() {
    dotenv().ok();

    let mut stockframe = StockFrame::new(Some(vec![String::from("AAPL"), String::from("TSLA")]));
    let mut _symbol_groups = stockframe.update_symbol_groups();

    stockframe.parse_dt_column();
    stockframe.fill_date_range();
    stockframe.fill_nulls();

    unsafe {
        stockframe.calc_technical_indicators();
        stockframe.frame = Box::new(stockframe.frame.fill_null(FillNullStrategy::Zero).unwrap());
        _symbol_groups = stockframe.update_symbol_groups();
    }

    stockframe.clean();
    println!("{:?}", stockframe.frame);
}
