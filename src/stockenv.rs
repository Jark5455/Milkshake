extern crate polars;

use crate::environment::{Environment, Restart, Spec, Terminate, Trajectory, Transition};
use crate::stockframe::StockFrame;

#[derive(Clone)]
pub struct StockEnv {
    pub stockframe: Box<StockFrame>,
    pub data: polars::prelude::DataFrame,

    pub iteration: u32,
    pub feature_length: u32,
    pub train_start: polars::export::chrono::NaiveDateTime,
    pub train_end: polars::export::chrono::NaiveDateTime,
    pub timestamp: polars::export::chrono::NaiveDateTime,
    pub episode_ended: bool,

    pub timeline: Vec<polars::export::chrono::NaiveDateTime>,
    pub acc_balance: Vec<f64>,
    pub total_asset: Vec<f64>,
    pub portfolio_asset: Vec<f64>,
    pub buy_price: Vec<f64>,
    pub unrealized_pnl: Vec<f64>,

    pub portfolio_value: f64,
    pub state: Vec<f64>,
    pub reward: f64,
}

fn calc_returns(series: polars::prelude::Series) -> polars::prelude::Series {
    let period_return = series.clone() / series.clone().shift(1) - 1;
    period_return.slice(1, period_return.len()).clone()
}

fn calc_gain_to_pain(series: polars::prelude::Series) -> f64 {
    let returns = calc_returns(series.drop_nulls().tail(Some(30)));
    let sum_returns: f64 = returns.sum().unwrap();

    let sum_neg_returns: f64 = returns
        .iter()
        .map(|f| {
            let x = match f {
                polars::prelude::AnyValue::Float64(z) => z,
                _ => 0f64,
            };

            if x < 0f64 {
                x.abs()
            } else {
                0f64
            }
        })
        .collect::<Vec<f64>>()
        .iter()
        .sum();

    sum_returns / (sum_neg_returns + 1f64)
}

fn calc_lake_ratio(series: polars::prelude::Series) -> f64 {
    let mut water = 0f64;
    let mut earth = 0f64;
    let mut peak: f64;
    let mut waterlevel: Vec<f64> = vec![];
    let mut s = calc_returns(series.clone());
    s = s
        .add_to(&<polars::prelude::Series as polars::prelude::NamedFrom<
            Vec<f64>,
            _,
        >>::new("_", vec![1f64; s.len()]))
        .unwrap();
    s = polars::prelude::cum_prod(&s, false).unwrap().drop_nulls();

    for (idx, f) in s.iter().enumerate() {
        let x = match f {
            polars::prelude::AnyValue::Float64(z) => z,
            _ => 0f64,
        };

        if idx == 0 {
            peak = x;
        } else {
            peak = s.slice(0, idx).max().unwrap().unwrap();
        }

        waterlevel.push(peak);

        if (x) < peak {
            water += peak - x
        }

        earth += x;
    }

    water / earth
}

impl Environment for StockEnv {
    fn action_spec(&self) -> Spec {
        Spec {
            min: -1.0,
            max: 1.0,
            shape: tickers.len() as u32,
        }
    }

    fn observation_spec(&self) -> Spec {
        Spec {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            shape: self.state.len() as u32,
        }
    }

    fn step(&mut self, action: Vec<f64>) -> Box<dyn Trajectory> {
        if self.episode_ended {
            return self.reset();
        }

        let mut new_ts = self.timestamp + polars::export::chrono::Duration::minutes(1);
        let mut data: polars::prelude::DataFrame;

        loop {
            data = polars::prelude::IntoLazy::lazy(self.stockframe.frame.borrow().clone())
                .filter(
                    polars::prelude::col("timestamp")
                        .dt()
                        .timestamp(polars::datatypes::TimeUnit::Milliseconds)
                        .eq(new_ts.timestamp_millis()),
                )
                .collect()
                .unwrap();

            if data.shape().0 != 0 {
                break;
            } else {
                new_ts += polars::export::chrono::Duration::minutes(1);

                if new_ts.timestamp_millis() > self.train_end.timestamp_millis() {
                    self.episode_ended = true;
                    return Box::new(Terminate {
                        observation: self.state.clone(),
                        reward: 0.0,
                    });
                }
            }
        }

        let flat_data: Vec<f64> = data
            .clone()
            .drop_many(&[String::from("symbol"), String::from("timestamp")])
            .to_ndarray::<polars::prelude::Float64Type>(polars::prelude::IndexOrder::C)
            .unwrap()
            .iter()
            .copied()
            .collect();
        self.data = data.clone();
        self.timestamp = new_ts;

        self.portfolio_value = (0..tickers.len())
            .collect::<Vec<usize>>()
            .iter()
            .map(|idx| {
                let symbol = tickers[*idx];
                let df = self.data.clone();
                let ticker_df = polars::prelude::IntoLazy::lazy(df)
                    .filter(polars::prelude::col("symbol").eq(polars::prelude::lit(symbol)))
                    .collect()
                    .unwrap();

                assert_ne!(ticker_df.shape().0, 0); // data must exist nulls are bad

                ticker_df["close"].f64().unwrap().get(0).unwrap()
            })
            .collect::<Vec<f64>>()
            .iter()
            .sum();

        let total_asset_starting = self.state[0] + self.portfolio_value;

        // we do all the sell order before buy orders to free up cash
        let mut indices: Vec<usize> = (0..action.len()).collect();
        indices.sort_by(|&i, &j| action[i].partial_cmp(&action[j]).unwrap());

        for idx in indices {
            if action[idx] < 0f64 {
                self.sell(idx as u32, action[idx]);
            } else if action[idx] > 0f64 {
                self.buy(idx as u32, action[idx]);
            }
        }

        self.unrealized_pnl = (0..tickers.len())
            .collect::<Vec<usize>>()
            .iter()
            .map(|idx| {
                let symbol = tickers[*idx];
                let df = self.data.clone();
                let ticker_df = polars::prelude::IntoLazy::lazy(df)
                    .filter(polars::prelude::col("symbol").eq(polars::prelude::lit(symbol)))
                    .collect()
                    .unwrap();

                assert_ne!(ticker_df.shape().0, 0); // data must exist nulls are bad

                (ticker_df["close"].f64().unwrap().get(0).unwrap() - self.buy_price[*idx])
                    * self.state[*idx + self.feature_length as usize]
            })
            .collect::<Vec<f64>>();

        self.state = [
            vec![self.state[0]],
            self.unrealized_pnl.clone(),
            flat_data,
            self.state[(self.feature_length as usize)..].to_vec(),
        ]
        .concat();

        self.portfolio_value = (0..tickers.len())
            .collect::<Vec<usize>>()
            .iter()
            .map(|idx| {
                let symbol = tickers[*idx];
                let df = self.data.clone();
                let ticker_df = polars::prelude::IntoLazy::lazy(df)
                    .filter(polars::prelude::col("symbol").eq(polars::prelude::lit(symbol)))
                    .collect()
                    .unwrap();

                assert_ne!(ticker_df.shape().0, 0); // data must exist nulls are bad

                ticker_df["close"].f64().unwrap().get(0).unwrap()
            })
            .collect::<Vec<f64>>()
            .iter()
            .sum();

        let total_asset_ending = self.state[0] + self.portfolio_value;

        self.acc_balance.push(self.state[0]);
        self.portfolio_asset.push(self.portfolio_value);
        self.total_asset.push(total_asset_ending);
        self.timeline.push(self.timestamp);

        if self.total_asset.len() > 29 {
            let total_asset = <polars::prelude::Series as polars::prelude::NamedFrom<
                Vec<f64>,
                _,
            >>::new("_", self.total_asset.clone());
            self.reward = total_asset_ending - total_asset_starting
                + (100f64 * calc_gain_to_pain(total_asset.clone()))
                - (500f64 * calc_lake_ratio(total_asset.clone()));
        } else {
            self.reward = total_asset_ending - total_asset_starting;
        }

        Box::new(Transition {
            observation: self.state.clone(),
            reward: self.reward,
        })
    }

    fn reset(&mut self) -> Box<dyn Trajectory> {
        self.episode_ended = false;
        self.acc_balance = vec![10000f64];
        self.total_asset = vec![10000f64];
        self.portfolio_asset = vec![0f64];
        self.buy_price = vec![0f64; tickers.len()];
        self.unrealized_pnl = vec![0f64];
        self.portfolio_value = 0.0;

        self.timestamp = self.train_start;
        self.timeline = vec![self.timestamp];

        self.data = polars::prelude::IntoLazy::lazy(self.stockframe.frame.borrow().clone())
            .filter(
                polars::prelude::col("timestamp")
                    .dt()
                    .timestamp(polars::datatypes::TimeUnit::Milliseconds)
                    .eq(self.timestamp.timestamp_millis()),
            )
            .collect()
            .unwrap();

        let flat_data: Vec<f64> = self
            .data
            .clone()
            .drop_many(&[String::from("symbol"), String::from("timestamp")])
            .to_ndarray::<polars::prelude::Float64Type>(polars::prelude::IndexOrder::C)
            .unwrap()
            .iter()
            .copied()
            .collect();
        self.state = [
            self.acc_balance.clone(),
            self.unrealized_pnl.clone(),
            flat_data,
            vec![0f64; tickers.len()],
        ]
        .concat();
        self.iteration += 1;

        Box::new(Restart {
            observation: self.state.clone(),
        })
    }
}

// I selected these from s&p 500 index but didnt want these to be all tech stocks so I hand picked them, gotta have some portfolio diversity
const tickers: [&str; 25] = [
    "AAPL", "AMD", "AMGN", "BA", "BAC", "BRK.B", "COST", "CRM", "DIS", "GOOG", "JNJ", "JPM", "MA",
    "MRK", "MSFT", "NKE", "NVDA", "PEP", "RTX", "SPY", "TSLA", "UNH", "UPS", "V", "WMT",
];

impl StockEnv {
    pub fn new(
        start: polars::export::chrono::NaiveDateTime,
        end: polars::export::chrono::NaiveDateTime,
    ) -> Self {
        let mut stockframe = StockFrame::new(
            Some(tickers.iter().map(|s| String::from(*s)).collect()),
            Some(start),
            Some(end),
        );

        stockframe.parse_dt_column();
        stockframe.fill_date_range();
        stockframe.fill_nulls();

        unsafe {
            stockframe.calc_technical_indicators();
        }

        // fill volume, vwap, and trade_count with zeros
        stockframe.frame = std::cell::RefCell::new(
            stockframe
                .clone()
                .frame
                .borrow_mut()
                .fill_null(polars::prelude::FillNullStrategy::Zero)
                .unwrap(),
        );
        stockframe.clean();

        // sort
        stockframe.update_symbol_groups();
        stockframe.frame = std::cell::RefCell::new(
            stockframe
                .clone()
                .frame
                .borrow_mut()
                .sort(["symbol", "timestamp"], vec![false, false], false)
                .unwrap(),
        );

        let acc_balance = vec![10000f64];
        let total_asset = vec![10000f64];
        let portfolio_asset = vec![0f64];
        let buy_price = vec![0f64; tickers.len()];
        let unrealized_pnl = vec![0f64];

        let mut df_start = stockframe.get_min_timestamp();
        let df_end = stockframe.get_min_timestamp();

        // search for next valid timestamp
        let mut data: polars::prelude::DataFrame;

        loop {
            data = polars::prelude::IntoLazy::lazy(stockframe.frame.borrow().clone())
                .filter(
                    polars::prelude::col("timestamp")
                        .dt()
                        .timestamp(polars::datatypes::TimeUnit::Milliseconds)
                        .eq(df_start.timestamp_millis()),
                )
                .collect()
                .unwrap();

            if data.shape().0 != 0 {
                break;
            } else {
                df_start += polars::export::chrono::Duration::minutes(1);
            }
        }

        let timeline = vec![df_start];

        let flat_data: Vec<f64> = data
            .clone()
            .drop_many(&[String::from("symbol"), String::from("timestamp")])
            .to_ndarray::<polars::prelude::Float64Type>(polars::prelude::IndexOrder::C)
            .unwrap()
            .iter()
            .copied()
            .collect();
        let feature_length = 2 + flat_data.len();

        StockEnv {
            stockframe: Box::new(stockframe),
            iteration: 0,
            feature_length: feature_length as u32,
            train_start: df_start,
            train_end: df_end,
            timestamp: df_start,
            episode_ended: true,
            timeline: timeline.clone(),
            acc_balance: acc_balance.clone(),
            total_asset: total_asset.clone(),
            portfolio_asset: portfolio_asset.clone(),
            buy_price: buy_price.clone(),
            unrealized_pnl: unrealized_pnl.clone(),
            portfolio_value: 0.0,
            data: data.clone(),
            state: [
                acc_balance,
                unrealized_pnl,
                flat_data,
                vec![0f64; tickers.len()],
            ]
            .concat(),
            reward: 0.0,
        }
    }

    pub fn buy(&mut self, idx: u32, action: f64) {
        let symbol = tickers[idx as usize];
        let df = self.data.clone();
        let ticker_df = polars::prelude::IntoLazy::lazy(df)
            .filter(polars::prelude::col("symbol").eq(polars::prelude::lit(symbol)))
            .collect()
            .unwrap();

        assert_ne!(ticker_df.shape().0, 0); // data must exist nulls are bad

        let price = ticker_df["close"].f64().unwrap().get(0).unwrap();
        let available_unit = (self.state[0] / price).floor();
        let num_share = (action * available_unit).floor();

        self.state[0] -= num_share * price;

        // if theres existing holdings take average price
        if self.state[(idx + self.feature_length) as usize] > 0f64 {
            let existing_holdings = self.state[(idx + self.feature_length) as usize];
            let previous_buy_price = self.buy_price[idx as usize];
            let new_holding = existing_holdings + num_share;
            self.buy_price[idx as usize] =
                ((existing_holdings * previous_buy_price) + (price * num_share)) / new_holding;
        } else if self.state[(idx + self.feature_length) as usize] == 0.0 {
            self.buy_price[idx as usize] = price;
        }

        self.state[(idx + self.feature_length) as usize] += num_share;
    }

    pub fn sell(&mut self, idx: u32, action: f64) {
        let num_share = (action.abs() * self.state[(idx + self.feature_length) as usize]).floor();

        let symbol = tickers[idx as usize];
        let df = self.data.clone();
        let ticker_df = polars::prelude::IntoLazy::lazy(df)
            .filter(polars::prelude::col("symbol").eq(polars::prelude::lit(symbol)))
            .collect()
            .unwrap();

        assert_ne!(ticker_df.shape().0, 0); // data must exist nulls are bad

        let price = ticker_df["close"].f64().unwrap().get(0).unwrap();

        if self.state[(idx + self.feature_length) as usize] > 0f64 {
            self.state[0] += price * num_share;
            self.state[(idx + self.feature_length) as usize] -= num_share;

            // reset price if thats the last share
            if self.state[(idx + self.feature_length) as usize] == 0f64 {
                self.buy_price[idx as usize] = 0.0;
            }
        }
    }
}
