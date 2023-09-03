#![allow(nonstandard_style)]
#![allow(dead_code)]
mod stockframe;

use dotenv::dotenv;
use stockframe::StockFrame;

fn main() {
    dotenv().ok();

    let mut stockframe = StockFrame::new(Some(vec![String::from("AAPL"), String::from("TSLA")]));
    let _symbol_groups = stockframe.update_symbol_groups();

    stockframe.parse_dt_column();
    stockframe.fill_date_range();
    stockframe.fill_nulls();

    unsafe {
        stockframe.calc_technical_indicators();
    }

    print!("{:?}", stockframe.frame);
}
