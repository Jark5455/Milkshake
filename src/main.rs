#![allow(nonstandard_style)]
mod stockframe;

use dotenv::dotenv;
use stockframe::StockFrame;

fn main() {
    dotenv().ok();

    let stockframe = StockFrame::new(None);
    let symbol_groupby = stockframe.frame.groupby(["symbol"]).unwrap();

    print!("{:?}", stockframe.frame);
}
