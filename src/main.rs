mod stockframe;

use dotenv::dotenv;
use std::env;
fn main() {
    dotenv().ok();

    const ALPACA_KEY: String = env::var("ALPACA_KEY").expect("Alpaca Key environment variable not set");
    const ALPACA_SECRET: String = env::var("ALPACA_SECRET").expect("Alpaca Secret environment variable not set");

    for (key, value) in env::vars() {
        println!("{}: {}", key, value);
    }
}
