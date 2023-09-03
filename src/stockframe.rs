use anyhow::{Error, Result};
use curl::easy::{Easy, List};
use polars::prelude::{DataFrame, JsonReader, SerReader, IntoLazy, StrptimeOptions, DataType, TimeUnit, Series, NamedFrom, TakeRandom, DataFrameJoinOps, UniqueKeepStrategy, GroupBy};
use polars::export::chrono::{Utc, Duration, NaiveDateTime, SecondsFormat, TimeZone};
use polars::prelude::{col, count, lit};
use serde_json::{Value};
use std::{env, io::Cursor, mem, ops::Sub};
use libc::{calloc, c_int, close};
use ta_lib_sys::MAType;
use ta_lib_sys::RetCode::{INTERNAL_ERROR, SUCCESS};

pub(crate) struct StockFrame {
    pub columns: Vec<String>,
    pub tickers: Vec<String>,
    pub frame: Box<DataFrame>,
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
                    lit(ticker.as_str()).alias("symbol")
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

        let columns_list: Vec<String> = vec!["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap", "trade_count", "adx", "atr", "aroonosc", "aroonu", "aroond", "bband_up", "bband_mid", "bband_low", "macd", "macdsignal", "macdhist", "rsi", "stoch_slowk", "stoch_slowd", "sma"].iter().map(|s| String::from(*s)).collect();
        let tickers_list = tickers.unwrap();

        let end = Utc::now().date_naive().and_hms_micro_opt(0, 0, 0, 0).unwrap();
        let start = end.sub(Duration::days(30));

        let dataframe = StockFrame::grab_latest_data(start, end, &tickers_list).lazy().with_columns(
            columns_list[9..].iter().map(|s| lit(polars::prelude::NULL).alias(s)).collect::<Vec<_>>().as_slice()
        ).collect().unwrap().select(&columns_list).unwrap().to_owned();

        let row_count = dataframe.clone().lazy().select([count().alias("count")]).collect().unwrap().column("count").unwrap().u32().unwrap().get(0).unwrap();
        assert!(row_count > 0);

        let dataframe_box = Box::new(dataframe.clone());

        StockFrame {
            columns: columns_list,
            tickers: tickers_list,
            frame: dataframe_box,
            len: row_count
        }
    }

    pub(crate) fn parse_dt_column(&mut self) {
        let lazy_df = self.frame.as_ref().clone().lazy();

        let new_df = lazy_df.with_columns([
            col("timestamp").str().strptime(DataType::Datetime(TimeUnit::Milliseconds, None), StrptimeOptions {
                format: Some("%+".into()),
                strict: false,
                exact: true,
                cache: false,
                use_earliest: None,
            }).alias("timestamp")
        ]).collect().expect("Failed to parse date time index");

        let _ = std::mem::replace(self.frame.as_mut(), new_df.clone());
    }

    pub(crate) fn fill_date_range(&mut self) {
        let df = self.frame.as_ref().clone();

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

        let ts_range: Vec<NaiveDateTime> = date_range.iter().map(|ts| Utc.timestamp_opt(ts / 1000, 0).unwrap().naive_utc()).collect();
        let ts_range_series = Series::new("timestamp", ts_range);
        let new_rows = DataFrame::new(vec![ts_range_series]).unwrap().lazy();

        let lazy_df = df.clone().lazy();
        let symbol_df = lazy_df.select([col("symbol")]).unique(None, UniqueKeepStrategy::First);
        let new_index = symbol_df.cross_join(new_rows).collect().unwrap();

        let new_df = df.clone().outer_join(&new_index, ["symbol", "timestamp"], ["symbol", "timestamp"]).unwrap();
        let _ = std::mem::replace(self.frame.as_mut(), new_df.clone());
    }

    pub(crate) fn fill_nulls(&mut self) {
        let df = self.frame.as_ref().clone();

        let ffill_df = df.lazy().with_columns([
            col("close").forward_fill(None).backward_fill(None).over(["symbol"])
        ]).collect().unwrap();

        let new_df = ffill_df.lazy().with_columns([
            col("open").fill_null(col("close")),
            col("high").fill_null(col("close")),
            col("low").fill_null(col("close"))
        ]).collect().unwrap();

        let _ = std::mem::replace(self.frame.as_mut(), new_df.clone());
    }

    pub(crate) fn update_symbol_groups(&mut self) -> Box<GroupBy> {
        return Box::new(self.frame.groupby(["symbol"]).unwrap())
    }
    pub(crate) unsafe fn calc_technical_indicators(&mut self) {
        // force sort by symbol
        let symbol_groups = self.update_symbol_groups();

        for idx in symbol_groups.get_groups().clone().iter() {
            let symbol_df = symbol_groups.df.slice(idx.first() as i64, idx.len() - 1).clone();
            let high: Vec<f64> = symbol_df.column("high").unwrap().f64().unwrap().into_no_null_iter().collect();
            let low: Vec<f64> = symbol_df.column("low").unwrap().f64().unwrap().into_no_null_iter().collect();
            let close: Vec<f64> = symbol_df.column("close").unwrap().f64().unwrap().into_no_null_iter().collect();

            let mut s: c_int = 0;
            let mut n: c_int = 0;

            let adx_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let atr_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let aroon_up_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let aroon_down_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let bband_up_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let bband_mid_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let bband_low_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let macd_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let macdsignal_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let macdhist_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let rsi_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let stoch_slowk_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let stoch_slowd_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;
            let sma_ptr = calloc(mem::size_of::<f64>(), idx.len()) as *mut f64;

            let mut status = INTERNAL_ERROR;
            status = ta_lib_sys::ADX(0, (close.len() - 1) as c_int, high.as_ptr(), low.as_ptr(), close.as_ptr(), 14, &mut s as *mut c_int, &mut n as *mut c_int, adx_ptr);
            assert_eq!(status, SUCCESS);
            let adx = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(adx_ptr, n as usize, idx.len())].concat();
            status = ta_lib_sys::ATR(0, (close.len() - 1) as c_int, high.as_ptr(), low.as_ptr(), close.as_ptr(), 14, &mut s as *mut c_int, &mut n as *mut c_int, atr_ptr);
            assert_eq!(status, SUCCESS);
            let atr = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(atr_ptr, n as usize, idx.len())].concat();
            status = ta_lib_sys::AROON(0, (close.len() - 1) as c_int, high.as_ptr(), low.as_ptr(), 14, &mut s as *mut c_int, &mut n as *mut c_int, aroon_down_ptr, aroon_up_ptr);
            assert_eq!(status, SUCCESS);
            let aroon_up = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(aroon_up_ptr, n as usize, idx.len())].concat();
            let aroon_down = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(aroon_down_ptr, n as usize, idx.len())].concat();
            status = ta_lib_sys::BBANDS(0, (close.len() - 1) as c_int, close.as_ptr(), 5, 2f64, 2f64, MAType::MAType_SMA, &mut s as *mut c_int, &mut n as *mut c_int, bband_up_ptr, bband_mid_ptr, bband_low_ptr);
            assert_eq!(status, SUCCESS);
            let bbands_low = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(bband_low_ptr, n as usize, idx.len())].concat();
            let bbands_mid = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(bband_mid_ptr, n as usize, idx.len())].concat();
            let bbands_up = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(bband_up_ptr, n as usize, idx.len())].concat();
            status = ta_lib_sys::MACD(0, (close.len() - 1) as c_int, close.as_ptr(), 12, 26, 9, &mut s as *mut c_int, &mut n as *mut c_int, macd_ptr, macdsignal_ptr, macdhist_ptr);
            assert_eq!(status, SUCCESS);
            let macd = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(macd_ptr, n as usize, idx.len())].concat();
            let macdhist = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(macdhist_ptr, n as usize, idx.len())].concat();
            let macdsignal = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(macdsignal_ptr, n as usize, idx.len())].concat();
            status = ta_lib_sys::RSI(0, (close.len() - 1) as c_int, close.as_ptr(), 14, &mut s as *mut c_int, &mut n as *mut c_int, rsi_ptr);
            assert_eq!(status, SUCCESS);
            let rsi = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(rsi_ptr, n as usize, idx.len())].concat();
            status = ta_lib_sys::STOCH(0, (close.len() - 1) as c_int, high.as_ptr(), low.as_ptr(), close.as_ptr(), 5, 3, MAType::MAType_SMA, 3, MAType::MAType_SMA, &mut s as *mut c_int, &mut n as *mut c_int, stoch_slowk_ptr, stoch_slowd_ptr);
            assert_eq!(status, SUCCESS);
            let stoch_slowd = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(stoch_slowd_ptr, n as usize, idx.len())].concat();
            let stoch_slowk = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(stoch_slowk_ptr, n as usize, idx.len())].concat();
            status = ta_lib_sys::SMA(0, (close.len() - 1) as c_int, close.as_ptr(), 30, &mut s as *mut c_int, &mut n as *mut c_int, sma_ptr);
            assert_eq!(status, SUCCESS);
            let sma = [vec![0f64; (s + 1) as usize], Vec::<f64>::from_raw_parts(sma_ptr, n as usize, idx.len())].concat();

            let adx_series = Series::new("adx", adx.iter());
            println!("{}", adx_series);
        }
    }

    pub(crate) fn clean(&mut self) {

    }
}