use crate::stockframe::StockFrame;
use crate::environment::Environment;
use polars::export::chrono::NaiveDateTime;

pub(crate) struct Spec {
    pub min: u32,
    pub max: u32,
    pub shape: ()
}
pub(crate) struct StockEnv {
    pub stockframe: Box<StockFrame>,

    pub iteration: u32,
    pub train_start: NaiveDateTime,
    pub train_end: NaiveDateTime,
    pub timestamp: NaiveDateTime,
    pub episode_ended: bool,

    pub timeline: Vec<NaiveDateTime>,
    pub acc_balance: Vec<f64>,
    pub total_asset: Vec<f64>,
    pub portfolio_asset: Vec<f64>,
    pub buy_price: Vec<f64>,
    pub unrealized_pnl: Vec<f64>,

    pub state: Vec<f64>,
    pub reward: f64
}

impl Environment for StockEnv {

}