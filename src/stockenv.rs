use crate::stockframe::StockFrame;
use crate::environment::Environment;
use polars::export::chrono::{Datelike, NaiveDateTime, Timelike};
use polars::prelude::{DataFrame, FillNullStrategy, Float64Type, IntoLazy, IndexOrder};

pub(crate) struct Spec {
    pub min: u32,
    pub max: u32,
    pub shape: u32
}
pub(crate) struct StockEnv {
    pub stockframe: Box<StockFrame>,
    pub data: DataFrame,

    pub iteration: u32,
    pub feature_length: u32,
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

    pub portfolio_value: f64,
    pub state: Vec<f64>,
    pub reward: f64
}

impl Environment for StockEnv {

}

// I selected these from s&p 500 index but didnt want these to be all tech stocks so I hand picked them, gotta have some portfolio diversity
const tickers: [&str; 25] = ["AAPL", "TSLA", "GOOG", "WMT", "NVDA", "MSFT", "NKE", "BRK.B", "UNH", "SPY", "JPM", "AMD", "UPS", "JNJ", "V", "MA", "PEP", "COST", "DIS", "BAC", "CRM", "MRK", "AMGN", "BA", "RTX"];

impl StockEnv {
    pub(crate) fn new(start: NaiveDateTime, end: NaiveDateTime) -> StockEnv {
        let mut stockframe = StockFrame::new(Some(tickers.iter().map(|s| String::from(*s)).collect()), Some(start.clone()), Some(end.clone()));
        let mut _symbol_groups = stockframe.update_symbol_groups();

        stockframe.parse_dt_column();
        stockframe.fill_date_range();
        stockframe.fill_nulls();

        unsafe {
            stockframe.calc_technical_indicators();
            stockframe.frame = Box::new(stockframe.frame.fill_null(FillNullStrategy::Zero).unwrap());
            _symbol_groups = stockframe.update_symbol_groups();
        }

        stockframe.frame.sort(&["symbol", "timestamp"], vec![true, false], false).unwrap();
        stockframe.clean();

        let timeline = vec![start.clone()];
        let acc_balance = vec![10000f64];
        let total_asset = vec![10000f64];
        let portfolio_asset = vec![0f64];
        let buy_price = vec![0f64; tickers.len()];
        let unrealized_pnl = vec![0f64];

        let feature_length = 2 + tickers.len() + stockframe.frame.get_columns().len();

        let data = stockframe.frame.clone().lazy().filter(
            polars::prelude::col("timestamp").dt().year().eq(start.year()).and(
                polars::prelude::col("timestamp").dt().month().eq(start.month()).and(
                    polars::prelude::col("timestamp").dt().day().eq(start.day()).and(
                        polars::prelude::col("timestamp").dt().day().eq(start.day()).and(
                            polars::prelude::col("timestamp").dt().hour().eq(start.time().hour()).and(
                                polars::prelude::col("timestamp").dt().minute().eq(start.time().minute()).and(
                                    polars::prelude::col("timestamp").dt().second().eq(start.time().second())
                                )
                            )
                        )
                    )
                )
            )
        ).collect().unwrap().drop_many(&[String::from("symbol"), String::from("timestamp")]);

        let flat_data: Vec<f64> = data.clone().to_ndarray::<Float64Type>(IndexOrder::C).unwrap().iter().map(|f: &f64| *f).collect();

        println!("data: {:?}", data);
        println!("flat data: {:?}", flat_data);

        return StockEnv {
            stockframe: Box::new(stockframe),
            iteration: 0,
            feature_length: feature_length as u32,
            train_start: start,
            train_end: end,
            timestamp: start,
            episode_ended: false,
            timeline: timeline.clone(),
            acc_balance: acc_balance.clone(),
            total_asset: total_asset.clone(),
            portfolio_asset: portfolio_asset.clone(),
            buy_price: buy_price.clone(),
            unrealized_pnl: unrealized_pnl.clone(),
            portfolio_value: 0.0,
            data: data.clone(),
            state: [acc_balance, unrealized_pnl, flat_data, vec![0f64; tickers.len()]].concat(),
            reward: 0.0
        }
    }
}