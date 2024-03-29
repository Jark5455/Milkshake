extern crate anyhow;
extern crate curl;
extern crate serde_json;

// Helper class that constructs Dataframe for me
// in order to use it you must have alpaca api keys set as env variables
// ALPACA_KEY={your api key}
// ALPACA_SECRET={your secret key}

#[derive(Clone)]
pub struct StockFrame {
    pub columns: Vec<String>,
    pub tickers: Vec<String>,
    pub frame: std::cell::RefCell<polars::prelude::DataFrame>,
}

impl StockFrame {
    fn grab_entire_json(
        ticker: &String,
        uri: &String,
        page_token: Option<String>,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let alpaca_key: String = std::env::var("ALPACA_KEY")?;
        let alpaca_secret: String = std::env::var("ALPACA_SECRET")?;

        let mut easy = curl::easy::Easy::new();

        let mut data = Vec::new();
        let mut headers = curl::easy::List::new();

        headers.append(format!("APCA-API-KEY-ID: {}", alpaca_key).as_str())?;
        headers.append(format!("APCA-API-SECRET-KEY: {}", alpaca_secret).as_str())?;

        match page_token {
            None => easy.url(uri.as_str()),
            Some(token) => easy.url(format!("{}&page_token={}", uri, token).as_str()),
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
        let json_object: serde_json::Value = serde_json::from_str(json_string.as_str())?;

        match json_object.get("next_page_token") {
            None => anyhow::bail!(format!("Invalid API Response: {}", json_string)),

            Some(next_page_token) => {
                let mut bars = match json_object.get("bars") {
                    None => anyhow::bail!(format!("Invalid API Response: {}", json_string)),
                    Some(bars) => match bars.as_array() {
                        None => {
                            anyhow::bail!(format!("Failed to cast bars to array: {}", json_string))
                        }
                        Some(checked_bars) => {
                            Ok::<Vec<serde_json::Value>, anyhow::Error>(checked_bars.clone())
                        }
                    },
                }?;

                match next_page_token.as_str() {
                    None => Ok(bars),

                    Some(page_token) => {
                        bars.append(&mut StockFrame::grab_entire_json(
                            ticker,
                            uri,
                            Some(page_token.parse()?),
                        )?);

                        Ok(bars)
                    }
                }
            }
        }
    }

    fn grab_latest_data(
        start: polars::export::chrono::NaiveDateTime,
        end: polars::export::chrono::NaiveDateTime,
        tickers: &Vec<String>,
    ) -> polars::prelude::DataFrame {
        let mut df = polars::prelude::DataFrame::default();

        for ticker in tickers {
            let grab_ticker_data = || -> Result<std::io::Cursor<String>, anyhow::Error> {
                let uri = format!(
                    "https://data.alpaca.markets/v2/stocks/{}/bars?{}{}{}",
                    ticker,
                    format!(
                        "start={}&",
                        start
                            .and_utc()
                            .to_rfc3339_opts(polars::export::chrono::SecondsFormat::Secs, true)
                    ),
                    format!(
                        "end={}&",
                        end.and_utc()
                            .to_rfc3339_opts(polars::export::chrono::SecondsFormat::Secs, true)
                    ),
                    "timeframe=1Min"
                );
                let entire_json = StockFrame::grab_entire_json(ticker, &uri, None)?;
                let json_tree = serde_json::to_string(&entire_json)?;
                Ok(std::io::Cursor::new(json_tree))
            };

            let cursor = match grab_ticker_data() {
                Ok(cursor) => cursor,
                Err(err) => {
                    println!("Failed to grab data for ticker: {}, Error: {}", ticker, err);
                    continue;
                }
            };

            let mut tmp_df = <polars::prelude::DataFrame as polars::prelude::IntoLazy>::lazy(
                <polars::prelude::JsonReader<std::io::Cursor<String>> as polars::prelude::SerReader<
                    std::io::Cursor<String>,
                >>::finish(
                    <polars::prelude::JsonReader<std::io::Cursor<String>> as polars::prelude::SerReader<
                        std::io::Cursor<String>,
                    >>::new(cursor),
                )
                .unwrap(),
            )
            .with_columns([polars::prelude::lit(ticker.as_str()).alias("symbol")])
            .collect()
            .unwrap();

            tmp_df
                .set_column_names(
                    vec![
                        "close",
                        "high",
                        "low",
                        "trade_count",
                        "open",
                        "timestamp",
                        "volume",
                        "vwap",
                        "symbol",
                    ]
                    .as_slice(),
                )
                .expect("Collumn number mismatch");
            df = df.vstack(&tmp_df).unwrap();

            // prevent rate limiting
            std::thread::sleep(core::time::Duration::from_secs(4));
        }

        df
    }

    pub fn new(
        mut tickers: Option<Vec<String>>,
        mut start: Option<polars::export::chrono::NaiveDateTime>,
        mut end: Option<polars::export::chrono::NaiveDateTime>,
    ) -> Self {
        if tickers.is_none() {
            tickers = Some(["AAPL", "TSLA"].iter().map(|s| String::from(*s)).collect());
        }

        if start.is_none() || end.is_none() {
            end = Some(
                polars::export::chrono::Utc::now()
                    .date_naive()
                    .and_hms_micro_opt(0, 0, 0, 0)
                    .unwrap(),
            );
            start = Some(end.unwrap() - polars::export::chrono::Duration::days(30));
        }

        assert!(tickers.is_some());

        let columns_list: Vec<String> = vec![
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "trade_count",
            "adx",
            "atr",
            "aroonosc",
            "aroonu",
            "aroond",
            "bband_up",
            "bband_mid",
            "bband_low",
            "macd",
            "macdsignal",
            "macdhist",
            "rsi",
            "stoch_slowk",
            "stoch_slowd",
            "sma",
        ]
        .iter()
        .map(|s| String::from(*s))
        .collect();
        let tickers_list = tickers.unwrap();

        let dataframe = <polars::prelude::DataFrame as polars::prelude::IntoLazy>::lazy(
            StockFrame::grab_latest_data(start.unwrap(), end.unwrap(), &tickers_list),
        )
        .with_columns(
            columns_list[9..]
                .iter()
                .map(|s| polars::prelude::lit(polars::prelude::NULL).alias(s))
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .collect()
        .unwrap()
        .select(&columns_list)
        .unwrap()
        .to_owned();

        let dataframe_box = std::cell::RefCell::new(dataframe);

        StockFrame {
            columns: columns_list,
            tickers: tickers_list,
            frame: dataframe_box,
        }
    }

    pub fn parse_dt_column(&mut self) {
        let lazy_df = <polars::prelude::DataFrame as polars::prelude::IntoLazy>::lazy(
            self.frame.borrow().clone(),
        );

        let strptimeoptions = polars::prelude::StrptimeOptions {
            format: Some("%+".into()),
            strict: true,
            exact: false,
            cache: false,
        };

        let new_df = lazy_df
            .with_columns([polars::prelude::col("timestamp").str().strptime(
                polars::prelude::DataType::Datetime(polars::prelude::TimeUnit::Milliseconds, None),
                strptimeoptions,
                polars::prelude::lit("1970-01-01T00:00:00+00:00"),
            )])
            .collect()
            .expect("Failed to parse date time index");

        self.frame.replace(new_df);
    }

    pub fn get_min_timestamp(&self) -> polars::export::chrono::NaiveDateTime {
        let df = self.frame.borrow().clone();

        let dt_column = df
            .select_series(["timestamp"])
            .expect("Failed to find column named \"timestamp\"");
        let dt_series = dt_column.get(0).expect("Failed to get timestamp column");
        let ts_column: Vec<i64> = dt_series
            .datetime()
            .unwrap()
            .as_datetime_iter()
            .map(|dt| dt.unwrap().timestamp_millis())
            .collect();

        let min = *ts_column.iter().min().unwrap();
        <polars::export::chrono::Utc as polars::export::chrono::TimeZone>::timestamp_opt(
            &polars::export::chrono::Utc,
            min / 1000,
            0,
        )
        .unwrap()
        .naive_utc()
    }

    pub fn get_max_timestamp(&self) -> polars::export::chrono::NaiveDateTime {
        let df = self.frame.borrow().clone();

        let dt_column = df
            .select_series(["timestamp"])
            .expect("Failed to find column named \"timestamp\"");
        let dt_series = dt_column.get(0).expect("Failed to get timestamp column");
        let ts_column: Vec<i64> = dt_series
            .datetime()
            .unwrap()
            .as_datetime_iter()
            .map(|dt| dt.unwrap().timestamp_millis())
            .collect();

        let max = *ts_column.iter().max().unwrap();
        <polars::export::chrono::Utc as polars::export::chrono::TimeZone>::timestamp_opt(
            &polars::export::chrono::Utc,
            max / 1000,
            0,
        )
        .unwrap()
        .naive_utc()
    }

    pub fn fill_date_range(&mut self) {
        let df = self.frame.borrow().clone();

        let max = self.get_max_timestamp().timestamp_millis();
        let mut min = self.get_min_timestamp().timestamp_millis();
        let mut date_range: Vec<i64> = vec![];

        while min <= max {
            date_range.push(min);
            min += 60000;
        }

        let ts_range: Vec<polars::export::chrono::NaiveDateTime> = date_range
            .iter()
            .map(|ts| {
                <polars::export::chrono::Utc as polars::export::chrono::TimeZone>::timestamp_opt(
                    &polars::export::chrono::Utc,
                    ts / 1000,
                    0,
                )
                .unwrap()
                .naive_utc()
            })
            .collect();

        let ts_range_series = <polars::prelude::Series as polars::prelude::NamedFrom<
            &[polars::export::chrono::NaiveDateTime],
            [polars::export::chrono::NaiveDateTime],
        >>::new("timestamp", ts_range.as_slice());
        let new_rows = <polars::prelude::DataFrame as polars::prelude::IntoLazy>::lazy(
            polars::prelude::DataFrame::new(vec![ts_range_series]).unwrap(),
        );

        let lazy_df = <polars::prelude::DataFrame as polars::prelude::IntoLazy>::lazy(df.clone());
        let symbol_df = lazy_df
            .select([polars::prelude::col("symbol")])
            .unique(None, polars::prelude::UniqueKeepStrategy::First);
        let new_index = symbol_df.cross_join(new_rows).collect().unwrap();

        let new_df = <polars::prelude::DataFrame as polars::prelude::DataFrameJoinOps>::outer_join(
            &df.clone(),
            &new_index,
            ["symbol", "timestamp"],
            ["symbol", "timestamp"],
        )
        .unwrap();

        self.frame.replace(new_df);
    }

    pub fn fill_nulls(&mut self) {
        let lazy_df = <polars::prelude::DataFrame as polars::prelude::IntoLazy>::lazy(
            self.frame.borrow().clone(),
        );

        let ffill_df = lazy_df
            .with_columns([polars::prelude::col("close")
                .forward_fill(None)
                .backward_fill(None)
                .over(["symbol"])])
            .collect()
            .unwrap();

        let new_df = <polars::prelude::DataFrame as polars::prelude::IntoLazy>::lazy(ffill_df)
            .with_columns([
                polars::prelude::col("open").fill_null(polars::prelude::col("close")),
                polars::prelude::col("high").fill_null(polars::prelude::col("close")),
                polars::prelude::col("low").fill_null(polars::prelude::col("close")),
            ])
            .collect()
            .unwrap();

        self.frame.replace(new_df);
    }

    pub fn update_symbol_groups(&mut self) -> Box<polars::prelude::GroupBy> {
        return Box::new(self.frame.get_mut().group_by(["symbol"]).unwrap());
    }

    // bad TA-Lib wrapper
    pub unsafe fn calc_technical_indicators(&mut self) {
        // force sort by symbol
        let mut concat_df = polars::prelude::DataFrame::default();
        let columns = self.columns.clone();
        let symbol_groups = self.update_symbol_groups();

        for idx in symbol_groups.get_groups().clone().iter() {
            let symbol_df = symbol_groups
                .df
                .slice(idx.first() as i64, idx.len())
                .clone();
            let high: Vec<f64> = symbol_df
                .column("high")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect();
            let low: Vec<f64> = symbol_df
                .column("low")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect();
            let close: Vec<f64> = symbol_df
                .column("close")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect();

            let mut s: libc::c_int = 0;
            let mut n: libc::c_int = 0;

            let mut adx = vec![0f64; idx.len()];
            let mut atr = vec![0f64; idx.len()];
            let mut aroon_up = vec![0f64; idx.len()];
            let mut aroon_down = vec![0f64; idx.len()];
            let mut aroonosc = vec![0f64; idx.len()];
            let mut bband_up = vec![0f64; idx.len()];
            let mut bband_mid = vec![0f64; idx.len()];
            let mut bband_low = vec![0f64; idx.len()];
            let mut macd = vec![0f64; idx.len()];
            let mut macdsignal = vec![0f64; idx.len()];
            let mut macdhist = vec![0f64; idx.len()];
            let mut rsi = vec![0f64; idx.len()];
            let mut stoch_slowk = vec![0f64; idx.len()];
            let mut stoch_slowd = vec![0f64; idx.len()];
            let mut sma = vec![0f64; idx.len()];

            assert_eq!(
                crate::wrappers::talib::TA_ADX(
                    0,
                    (close.len() - 1) as libc::c_int,
                    high.as_ptr(),
                    low.as_ptr(),
                    close.as_ptr(),
                    14,
                    &mut s as *mut libc::c_int,
                    &mut n as *mut libc::c_int,
                    adx.as_mut_ptr(),
                ),
                crate::wrappers::talib::TA_RetCode_TA_SUCCESS
            );

            assert_eq!(
                crate::wrappers::talib::TA_ATR(
                    0,
                    (close.len() - 1) as libc::c_int,
                    high.as_ptr(),
                    low.as_ptr(),
                    close.as_ptr(),
                    14,
                    &mut s as *mut libc::c_int,
                    &mut n as *mut libc::c_int,
                    atr.as_mut_ptr(),
                ),
                crate::wrappers::talib::TA_RetCode_TA_SUCCESS
            );

            assert_eq!(
                crate::wrappers::talib::TA_AROON(
                    0,
                    (close.len() - 1) as libc::c_int,
                    high.as_ptr(),
                    low.as_ptr(),
                    14,
                    &mut s as *mut libc::c_int,
                    &mut n as *mut libc::c_int,
                    aroon_down.as_mut_ptr(),
                    aroon_up.as_mut_ptr(),
                ),
                crate::wrappers::talib::TA_RetCode_TA_SUCCESS
            );

            assert_eq!(
                crate::wrappers::talib::TA_AROONOSC(
                    0,
                    (close.len() - 1) as libc::c_int,
                    high.as_ptr(),
                    low.as_ptr(),
                    14,
                    &mut s as *mut libc::c_int,
                    &mut n as *mut libc::c_int,
                    aroonosc.as_mut_ptr(),
                ),
                crate::wrappers::talib::TA_RetCode_TA_SUCCESS
            );

            assert_eq!(
                crate::wrappers::talib::TA_BBANDS(
                    0,
                    (close.len() - 1) as libc::c_int,
                    close.as_ptr(),
                    5,
                    2f64,
                    2f64,
                    crate::wrappers::talib::TA_MAType_TA_MAType_SMA,
                    &mut s as *mut libc::c_int,
                    &mut n as *mut libc::c_int,
                    bband_up.as_mut_ptr(),
                    bband_mid.as_mut_ptr(),
                    bband_low.as_mut_ptr(),
                ),
                crate::wrappers::talib::TA_RetCode_TA_SUCCESS
            );

            assert_eq!(
                crate::wrappers::talib::TA_MACD(
                    0,
                    (close.len() - 1) as libc::c_int,
                    close.as_ptr(),
                    12,
                    26,
                    9,
                    &mut s as *mut libc::c_int,
                    &mut n as *mut libc::c_int,
                    macd.as_mut_ptr(),
                    macdsignal.as_mut_ptr(),
                    macdhist.as_mut_ptr(),
                ),
                crate::wrappers::talib::TA_RetCode_TA_SUCCESS
            );

            assert_eq!(
                crate::wrappers::talib::TA_RSI(
                    0,
                    (close.len() - 1) as libc::c_int,
                    close.as_ptr(),
                    14,
                    &mut s as *mut libc::c_int,
                    &mut n as *mut libc::c_int,
                    rsi.as_mut_ptr(),
                ),
                crate::wrappers::talib::TA_RetCode_TA_SUCCESS
            );

            assert_eq!(
                crate::wrappers::talib::TA_STOCH(
                    0,
                    (close.len() - 1) as libc::c_int,
                    high.as_ptr(),
                    low.as_ptr(),
                    close.as_ptr(),
                    5,
                    3,
                    crate::wrappers::talib::TA_MAType_TA_MAType_SMA,
                    3,
                    crate::wrappers::talib::TA_MAType_TA_MAType_SMA,
                    &mut s as *mut libc::c_int,
                    &mut n as *mut libc::c_int,
                    stoch_slowk.as_mut_ptr(),
                    stoch_slowd.as_mut_ptr(),
                ),
                crate::wrappers::talib::TA_RetCode_TA_SUCCESS
            );

            assert_eq!(
                crate::wrappers::talib::TA_SMA(
                    0,
                    (close.len() - 1) as libc::c_int,
                    close.as_ptr(),
                    30,
                    &mut s as *mut libc::c_int,
                    &mut n as *mut libc::c_int,
                    sma.as_mut_ptr(),
                ),
                crate::wrappers::talib::TA_RetCode_TA_SUCCESS
            );

            let mut new_df = symbol_df.clone();
            new_df = new_df.drop_many(columns[9..].as_ref());

            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("adx", adx),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("atr", atr),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("aroonosc", aroonosc),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("aroonu", aroon_up),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("aroond", aroon_down),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("bband_up", bband_up),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("bband_mid", bband_mid),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("bband_low", bband_low),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("macd", macd),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("macdsignal", macdsignal),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("macdhist", macdhist),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("rsi", rsi),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("stoch_slowk", stoch_slowk),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("stoch_slowd", stoch_slowd),
                )
                .unwrap();
            new_df
                .with_column(
                    <polars::prelude::Series as polars::prelude::NamedFromOwned<Vec<f64>>>::from_vec("sma", sma),
                )
                .unwrap();

            concat_df = concat_df.vstack(&new_df).unwrap();
        }

        self.frame.replace(concat_df);
    }

    // limit to trading hours (not including first 30 mins due to lack of data in that period)
    pub fn clean(&mut self) {
        let lazy = <polars::prelude::DataFrame as polars::prelude::IntoLazy>::lazy(
            self.frame.borrow().clone(),
        );

        let new_df = lazy
            .filter(
                polars::prelude::col("timestamp")
                    .dt()
                    .hour()
                    .lt_eq(polars::prelude::lit(20))
                    .and(
                        polars::prelude::col("timestamp")
                            .dt()
                            .hour()
                            .gt_eq(polars::prelude::lit(14)),
                    ),
            )
            .collect()
            .unwrap();

        self.frame.replace(new_df);
    }
}
