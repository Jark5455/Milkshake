#![allow(nonstandard_style)]
#![allow(dead_code)]
mod stockframe;

use dotenv::dotenv;
use stockframe::StockFrame;

fn main() {
    dotenv().ok();
    StockFrame::new(None);
}
