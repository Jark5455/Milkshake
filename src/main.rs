#![allow(nonstandard_style)]
#![allow(dead_code)]
mod stockframe;
mod environment;
mod stockenv;

use std::fs::File;
use dotenv::dotenv;
use polars::prelude::{CsvWriter, FillNullStrategy, SerWriter};
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

    let mut file = File::create("df.csv").expect("could not create file");

    CsvWriter::new(&mut file)
        .has_header(true)
        .with_delimiter(b',')
        .finish(stockframe.frame.as_mut()).expect("could write to file");
}
