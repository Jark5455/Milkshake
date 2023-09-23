#![allow(nonstandard_style)]
#![allow(dead_code)]

mod environment;
mod stockenv;
mod stockframe;
mod td3;
mod tests;

use dotenv::dotenv;
use std::ops::Deref;

fn main() {
    dotenv().ok();
}
