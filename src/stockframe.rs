use anyhow::{Error, Result};
use curl::easy::{Easy, List};
use polars::prelude::{DataFrame, JsonReader, SerReader, IntoLazy, StrptimeOptions, DataType, TimeUnit, Series, NamedFrom, TakeRandom, DataFrameJoinOps};
use polars::export::chrono::{Utc, Duration, NaiveDateTime, SecondsFormat, TimeZone, Timelike};
use serde_json::{Value};
use std::{env, io::Cursor, ops::Sub};
use polars::frame::UniqueKeepStrategy;

pub(crate) struct StockFrame {
    pub columns: Vec<String>,
    pub tickers: Vec<String>,
    pub frame: DataFrame,
    pub len: u32
}
impl StockFrame {
    fn grab_entire_json(ticker: &String, uri: &String, page_token: Option<String>) -> Result<Vec<Value>> {
        let alpaca_key: String = env::var("ALPACA_KEY")?;
        let alpaca_secret: String = env::var("ALPACA_SECRET")?;

        let mut easy = Easy::new();

        let mut data = Vec::new();
        let mut headers = List::new();

        headers.append(format!("APCA-API-KEY-ID: {}", alpaca_key).as_str())?;
        headers.append(format!("APCA-API-SECRET-KEY: {}", alpaca_secret).as_str())?;

        match page_token {
            None => { easy.url(uri.as_str()) }
            Some(token) => { easy.url(format!("{}&page_token={}", uri, token).as_str()) }
        }?;

        easy.http_headers(headers)?;

        {
            let mut transfer = easy.transfer();
            transfer.write_function(|new_data| {
                data.extend_from_slice(new_data);
                Ok(new_data.len())
            })?;

            transfer.perform()?;
        }

        let json_string = String::from_utf8(data)?;
        let json_object: Value = serde_json::from_str(json_string.as_str())?;

        match json_object.get("next_page_token") {
            None => {
                Err(Error::msg(format!("Invalid API Response: {}", json_string)))
            }

            Some(next_page_token) => {
                let mut bars = match json_object.get("bars") {
                    None => { Err(Error::msg(format!("Invalid API Response: {}", json_string))) }
                    Some(bars) => {
                        match bars.as_array() {
                            None => { Err(Error::msg(format!("Failed to cast bars to array: {}", json_string))) }
                            Some(checked_bars) => { Ok(checked_bars.clone()) }
                        }
                    }
                }?;

                match next_page_token.as_str() {
                    None => {
                        Ok(bars)
                    }

                    Some(page_token) => {
                        bars.append(&mut StockFrame::grab_entire_json(ticker, uri, Some(page_token.parse()?))?);

                        Ok(bars)
                    }
                }
            }
        }
    }
    fn grab_latest_data(start: NaiveDateTime, end: NaiveDateTime, tickers: &Vec<String>) -> DataFrame {
        let mut df = DataFrame::default();

        for ticker in tickers {

            let grab_ticker_data = || -> Result<Cursor<String>, Error> {
                let uri = format!("https://data.alpaca.markets/v2/stocks/{}/bars?{}{}{}", ticker, format!("start={}&", start.and_utc().to_rfc3339_opts(SecondsFormat::Secs, true)), format!("end={}&", end.and_utc().to_rfc3339_opts(SecondsFormat::Secs, true)), "timeframe=1Min");
                let entire_json = StockFrame::grab_entire_json(ticker, &uri, None)?;
                let json_tree = serde_json::to_string(&entire_json)?;
                Ok(Cursor::new(json_tree))
            };

            let cursor = match grab_ticker_data() {
                Ok(cursor) => { cursor }
                Err(err) => {
                    println!("Failed to grab data for ticker: {}, Error: {}", ticker, err.to_string());
                    continue;
                }
            };

            let mut tmp_df = JsonReader::new(cursor).finish().unwrap().lazy().with_columns(
                [
                    polars::prelude::lit(ticker.as_str()).alias("symbol")
                ]
            ).collect().unwrap();

            tmp_df.set_column_names(vec!["close", "high", "low", "trade_count", "open", "timestamp", "volume", "vwap", "symbol"].as_slice()).expect("Collumn number mismatch");
            df = df.vstack(&tmp_df).unwrap();

            std::thread::sleep(core::time::Duration::from_secs(4));
        }

        return df;
    }

    pub(crate) fn new(mut tickers: Option<Vec<String>>) -> StockFrame {
        if tickers.is_none() {
            tickers = Some(vec!["AAPL", "TSLA"].iter().map(|s| String::from(*s)).collect());
        }

        assert!(tickers.is_some());

        let columns_list: Vec<String> = vec!["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap", "trade_count", "adx", "aroonosc", "aroonu", "aroond", "bband_up", "bband_mid", "bband_low", "macd", "macdsignal", "macdhist", "rsi", "stoch_slowk", "stoch_slowd", "sma"].iter().map(|s| String::from(*s)).collect();
        let tickers_list = tickers.unwrap();

        let end = Utc::now().date_naive().and_hms_micro_opt(0, 0, 0, 0).unwrap();
        let start = end.sub(Duration::days(30));

        let dataframe = StockFrame::grab_latest_data(start, end, &tickers_list).lazy().with_columns(
            columns_list[9..].iter().map(|s| polars::prelude::lit(polars::prelude::NULL).alias(s)).collect::<Vec<_>>().as_slice()
        ).collect().unwrap().select(&columns_list).unwrap().to_owned();

        let row_count = dataframe.clone().lazy().select([polars::prelude::count().alias("count")]).collect().unwrap().column("count").unwrap().u32().unwrap().get(0).unwrap();
        assert!(row_count > 0);

        StockFrame {
            columns: columns_list,
            tickers: tickers_list,
            frame: dataframe,
            len: row_count
        }
    }

    pub(crate) fn parse_dt_column(&mut self) {
        let lazy_df = self.frame.clone().lazy();

        self.frame = lazy_df.with_columns([
            polars::prelude::col("timestamp").str().strptime(DataType::Datetime(TimeUnit::Milliseconds, None), StrptimeOptions {
                format: Some("%+".into()),
                strict: false,
                exact: true,
                cache: false,
                use_earliest: None,
            }).alias("timestamp")
        ]).collect().expect("Failed to parse date time index");
    }

    pub(crate) fn fill_date_range(&mut self) {
        let df = self.frame.clone();

        let dt_column = df.select_series(["timestamp"]).expect("Failed to find column named \"timestamp\"");
        let dt_series = dt_column.get(0).expect("Failed to get timestamp column");
        let ts_column: Vec<i64> = dt_series.datetime().unwrap().as_datetime_iter().map(|dt| dt.unwrap().timestamp_millis()).collect();

        let max = *ts_column.iter().max().unwrap();
        let mut min = *ts_column.iter().min().unwrap();
        let mut date_range: Vec<i64> = vec![];

        while min <= max {
            date_range.push(min);
            min += 60000;
        }

        let ts_range: Vec<NaiveDateTime> = date_range.iter().map(|ts| Utc.timestamp_opt(ts / 1000, 0).unwrap().naive_utc().with_nanosecond(0).unwrap()).collect();
        let ts_range_series = Series::new("timestamp", ts_range);
        let new_rows = DataFrame::new(vec![ts_range_series]).unwrap().lazy();

        let lazy_df = df.clone().lazy();
        let symbol_df = lazy_df.select([polars::prelude::col("symbol")]).unique(None, UniqueKeepStrategy::First);
        let new_index = symbol_df.cross_join(new_rows).collect().unwrap();

        let new_df = df.clone().outer_join(&new_index, ["symbol", "timestamp"], ["symbol", "timestamp"]).unwrap();
        self.frame = new_df
    }

    pub(crate) fn fill_nulls(&mut self) {

    }

    pub(crate) fn calc_technical_indicators(&mut self) {

    }

    pub(crate) fn clean(&mut self) {

    }
}