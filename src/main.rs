#![allow(nonstandard_style)]
#![allow(dead_code)]
mod stockframe;

use dotenv::dotenv;
use stockframe::StockFrame;

fn main() {
    dotenv().ok();

    let mut stockframe = StockFrame::new(None);
    let _symbol_groupby = stockframe.frame.groupby(["symbol"]).unwrap();

    stockframe.parse_dt_column();

    print!("{:?}", stockframe.frame);
}
