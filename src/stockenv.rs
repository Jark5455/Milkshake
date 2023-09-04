use crate::stockframe::StockFrame;
use crate::environment::{Environment, Restart, Spec, Trajectory, Transition};
use polars::export::chrono::{Datelike, Duration, NaiveDateTime, Timelike};
use polars::prelude::{DataFrame, FillNullStrategy, Float64Type, IntoLazy, IndexOrder, TakeRandom};

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
    fn action_spec(&self) -> Spec {
        return Spec {
            min: -1.0,
            max: 1.0,
            shape: tickers.len() as u32
        }
    }

    fn observation_spec(&self) -> Spec {
        return Spec {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            shape: self.state.len() as u32
        }
    }

    fn step(&mut self, action: Vec<f64>) -> Box<dyn Trajectory> {
        if self.episode_ended {
            return self.reset();
        }

        let mut new_ts = self.timestamp + Duration::minutes(1);
        let mut data: DataFrame;

        loop {
            data = self.stockframe.frame.clone().lazy().filter(
                polars::prelude::col("timestamp").dt().year().eq(new_ts.year()).and(
                    polars::prelude::col("timestamp").dt().month().eq(new_ts.month()).and(
                        polars::prelude::col("timestamp").dt().day().eq(new_ts.day()).and(
                            polars::prelude::col("timestamp").dt().day().eq(new_ts.day()).and(
                                polars::prelude::col("timestamp").dt().hour().eq(new_ts.time().hour()).and(
                                    polars::prelude::col("timestamp").dt().minute().eq(new_ts.time().minute()).and(
                                        polars::prelude::col("timestamp").dt().second().eq(new_ts.time().second())
                                    )
                                )
                            )
                        )
                    )
                )
            ).collect().unwrap();

            if data.shape().0 != 0 {
                break;
            } else {
                new_ts += Duration::minutes(1);
            }
        }

        let flat_data: Vec<f64> = data.clone().drop_many(&[String::from("symbol"), String::from("timestamp")]).to_ndarray::<Float64Type>(IndexOrder::C).unwrap().iter().map(|f: &f64| *f).collect();
        self.data = data.clone();
        self.timestamp = new_ts.clone();

        self.portfolio_value = vec![0..tickers.len()].iter().map(|idx: u32| {
            let symbol = tickers[idx as usize];
            let df = self.data.clone();
            let ticker_df = df.lazy().filter(
                polars::prelude::col("symbol").eq(polars::prelude::lit(symbol))
            ).collect().unwrap();

            assert_ne!(ticker_df.shape().0, 0); // data must exist nulls are bad

            ticker_df["close"].f64().unwrap().get(0).unwrap();
        }).collect().iter().sum();

        let total_asset_starting = self.state[0] + self.portfolio_value;

        // we do all the sell order before buy orders to free up cash
        let mut indices: Vec<usize> = (0..action.len()).collect();
        indices.sort_by_key(|&i| &data[i]);

        for idx in indices {
            if action[idx] < 0f64 {
                self.sell(idx as u32, action[idx]);
            } else if action[idx] > 0f64 {
                self.buy(idx as u32, action[idx]);
            }
        }

        self.unrealized_pnl = vec![0..tickers.len()].iter().map(|idx: u32| {
            let symbol = tickers[idx as usize];
            let df = self.data.clone();
            let ticker_df = df.lazy().filter(
                polars::prelude::col("symbol").eq(polars::prelude::lit(symbol))
            ).collect().unwrap();

            assert_ne!(ticker_df.shape().0, 0); // data must exist nulls are bad

            (ticker_df["close"].f64().unwrap().get(0).unwrap() - self.buy_price[idx]) * self.state[idx + self.feature_length];
        }).collect();

        self.state = [vec![self.state[0]], self.unrealized_pnl.clone(), flat_data, self.state[self.feature_length..]].concat();

        self.portfolio_value = vec![0..tickers.len()].iter().map(|idx: u32| {
            let symbol = tickers[idx as usize];
            let df = self.data.clone();
            let ticker_df = df.lazy().filter(
                polars::prelude::col("symbol").eq(polars::prelude::lit(symbol))
            ).collect().unwrap();

            assert_ne!(ticker_df.shape().0, 0); // data must exist nulls are bad

            ticker_df["close"].f64().unwrap().get(0).unwrap();
        }).collect().iter().sum();

        let total_asset_ending = self.state[0] + self.portfolio_value;

        self.acc_balance.push(self.state[0]);
        self.portfolio_asset.push(self.portfolio_value);
        self.total_asset.push(total_asset_ending);
        self.timeline.push(self.timestamp);

        if self.total_asset.len() > 29 {

        }

        return Box::new(Transition{ observation: self.state.clone(), reward: 0.0 })
    }
}

// I selected these from s&p 500 index but didnt want these to be all tech stocks so I hand picked them, gotta have some portfolio diversity
const tickers: [&str; 25] = ["AAPL", "AMD", "AMGN", "BA", "BAC", "BRK.B", "COST", "CRM", "DIS", "GOOG", "JNJ", "JPM", "MA", "MRK", "MSFT", "NKE", "NVDA", "PEP", "RTX", "SPY", "TSLA", "UNH", "UPS", "V", "WMT"];

impl StockEnv {
    pub(crate) fn new(start: NaiveDateTime, end: NaiveDateTime) -> StockEnv {
        let mut stockframe = StockFrame::new(Some(tickers.iter().map(|s| String::from(*s)).collect()), Some(start.clone()), Some(end.clone()));
        let mut _symbol_groups = stockframe.update_symbol_groups();

        stockframe.parse_dt_column();
        stockframe.fill_date_range();
        stockframe.fill_nulls();

        unsafe {
            stockframe.calc_technical_indicators();
        }

        // fill volume, vwap, and trade_count with zeros
        let _ = std::mem::replace(stockframe.frame.as_mut(), stockframe.frame.fill_null(FillNullStrategy::Zero).unwrap());
        _symbol_groups = stockframe.update_symbol_groups();
        stockframe.clean();

        // sort
        let _ = std::mem::replace(stockframe.frame.as_mut(), stockframe.frame.sort(&["symbol", "timestamp"], vec![false, false], false).unwrap());
        _symbol_groups = stockframe.update_symbol_groups();

        let acc_balance = vec![10000f64];
        let total_asset = vec![10000f64];
        let portfolio_asset = vec![0f64];
        let buy_price = vec![0f64; tickers.len()];
        let unrealized_pnl = vec![0f64];

        let mut df_start = stockframe.get_min_timestamp();
        let df_end = stockframe.get_min_timestamp();

        // search for next valid timestamp
        let mut data: DataFrame;

        loop {
            data = stockframe.frame.clone().lazy().filter(
                polars::prelude::col("timestamp").dt().year().eq(df_start.year()).and(
                    polars::prelude::col("timestamp").dt().month().eq(df_start.month()).and(
                        polars::prelude::col("timestamp").dt().day().eq(df_start.day()).and(
                            polars::prelude::col("timestamp").dt().day().eq(df_start.day()).and(
                                polars::prelude::col("timestamp").dt().hour().eq(df_start.time().hour()).and(
                                    polars::prelude::col("timestamp").dt().minute().eq(df_start.time().minute()).and(
                                        polars::prelude::col("timestamp").dt().second().eq(df_start.time().second())
                                    )
                                )
                            )
                        )
                    )
                )
            ).collect().unwrap();

            if data.shape().0 != 0 {
                break;
            } else {
                df_start += Duration::minutes(1);
            }
        }

        let timeline = vec![df_start.clone()];

        let flat_data: Vec<f64> = data.clone().drop_many(&[String::from("symbol"), String::from("timestamp")]).to_ndarray::<Float64Type>(IndexOrder::C).unwrap().iter().map(|f: &f64| *f).collect();
        let feature_length = 2 + flat_data.len();

        return StockEnv {
            stockframe: Box::new(stockframe),
            iteration: 0,
            feature_length: feature_length as u32,
            train_start: df_start.clone(),
            train_end: df_end.clone(),
            timestamp: df_start.clone(),
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

    fn reset(&mut self) -> Box<dyn Trajectory> {
        self.episode_ended = false;
        self.acc_balance = vec![10000f64];
        self.total_asset = vec![10000f64];
        self.portfolio_asset = vec![0f64];
        self.buy_price = vec![0f64; tickers.len()];
        self.unrealized_pnl = vec![0f64];
        self.portfolio_value = 0.0;

        self.timestamp = self.train_start.clone();
        self.timeline = vec![self.timestamp.clone()];

        self.data  = self.stockframe.frame.clone().lazy().filter(
            polars::prelude::col("timestamp").dt().year().eq(self.timestamp.year()).and(
                polars::prelude::col("timestamp").dt().month().eq(self.timestamp.month()).and(
                    polars::prelude::col("timestamp").dt().day().eq(self.timestamp.day()).and(
                        polars::prelude::col("timestamp").dt().day().eq(self.timestamp.day()).and(
                            polars::prelude::col("timestamp").dt().hour().eq(self.timestamp.time().hour()).and(
                                polars::prelude::col("timestamp").dt().minute().eq(self.timestamp.time().minute()).and(
                                    polars::prelude::col("timestamp").dt().second().eq(self.timestamp.time().second())
                                )
                            )
                        )
                    )
                )
            )
        ).collect().unwrap();

        let flat_data: Vec<f64> = self.data.clone().drop_many(&[String::from("symbol"), String::from("timestamp")]).to_ndarray::<Float64Type>(IndexOrder::C).unwrap().iter().map(|f: &f64| *f).collect();
        self.state = [self.acc_balance.clone(), self.unrealized_pnl.clone(), flat_data, vec![0f64; tickers.len()]].concat();
        self.iteration += 1;

        return Box::new(Restart{ observation: self.state.clone() })
    }

    pub(crate) fn buy(&mut self, idx: u32, action: f64) {
        let symbol = tickers[idx as usize];
        let df = self.data.clone();
        let ticker_df = df.lazy().filter(
            polars::prelude::col("symbol").eq(polars::prelude::lit(symbol))
        ).collect().unwrap();

        assert_ne!(ticker_df.shape().0, 0); // data must exist nulls are bad

        let price = ticker_df["close"].f64().unwrap().get(0).unwrap();
        let available_unit = (self.state[0] / price).floor();
        let num_share = (action * available_unit).floor();

        self.state[0] -= num_share * price;

        // if theres existing holdings take average price
        if self.state[idx + self.feature_length] > 0 {
            let existing_holdings = self.state[idx + self.feature_length];
            let previous_buy_price = self.buy_price[idx];
            let new_holding = existing_holdings + num_share;
            self.buy_price[idx] = ((existing_holdings * previous_buy_price) + (price * num_share)) / new_holding;
        } else if self.state[idx + self.feature_length] == 0.0 {
            self.buy_price[idx] = price;
        }

        self.state[idx + self.feature_length] += num_share;
    }

    pub(crate) fn sell(&mut self, idx: u32, action: f64) {
        let num_share = (action.abs() * self.state[idx + self.feature_length]).floor();

        let symbol = tickers[idx as usize];
        let df = self.data.clone();
        let ticker_df = df.lazy().filter(
            polars::prelude::col("symbol").eq(polars::prelude::lit(symbol))
        ).collect().unwrap();

        assert_ne!(ticker_df.shape().0, 0); // data must exist nulls are bad

        let price = ticker_df["close"].f64().unwrap().get(0).unwrap();

        if self.state[idx + self.feature_length] > 0  {
            self.state[0] += (price * num_share);
            self.state[idx + self.feature_length] -= num_share;

            // reset price if thats the last share
            if self.state[idx + self.feature_length] == 0.0 {
                self.buy_price[idx] = 0.0;
            }
        }
    }
}