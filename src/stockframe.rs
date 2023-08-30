use curl::easy::{Easy, List};
use polars::prelude::{DataFrame, JsonReader, SerReader, IntoLazy};
use serde_json::{Value};
use std::{env, io::Cursor, ops::Sub};
use time::{Duration, OffsetDateTime, format_description::well_known::Rfc3339};

pub(crate) struct StockFrame {
    pub columns: Vec<String>,
    pub tickers: Vec<String>,
    pub frame: DataFrame
}
impl StockFrame {
    fn grab_entire_json(ticker: &String, uri: &String) -> Vec<Value> {
        let alpaca_key: String = env::var("ALPACA_KEY").expect("Alpaca Key environment variable not set");
        let alpaca_secret: String = env::var("ALPACA_SECRET").expect("Alpaca Secret environment variable not set");

        let mut easy = Easy::new();

        let mut data = Vec::new();
        let mut headers = List::new();

        headers.append(format!("APCA-API-KEY-ID: {}", alpaca_key).as_str()).unwrap();
        headers.append(format!("APCA-API-SECRET-KEY: {}", alpaca_secret).as_str()).unwrap();

        easy.url(uri.as_str()).unwrap();
        easy.http_headers(headers).unwrap();

        {
            let mut transfer = easy.transfer();
            transfer.write_function(|new_data| {
                data.extend_from_slice(new_data);
                Ok(new_data.len())
            }).unwrap();

            transfer.perform().expect(format!("Failed to perform https request on uri: {}", uri).as_str());
        }

        let json_string = String::from_utf8(data).unwrap();
        let json_object: Value = serde_json::from_str(json_string.as_str()).expect(format!("Failed to parse json: {}", json_string).as_str());

        let bars = match json_object.get("next_page_token").expect(format!("Failed to read page token: {}", json_string).as_str()).as_str() {
            None => {
                match json_object.get("bars").unwrap().as_array() {
                    None => { Vec::new() }
                    Some(bars_sect) => { bars_sect.clone() }
                }
            }

            Some(next_page_token) => {
                let mut bars = json_object.get("bars").expect("Bars missing from json").as_array().unwrap().clone();
                let next_page_uri = &format!("https://data.alpaca.markets/v2/stocks/{}/bars?{}{}", ticker, format!("page_token={}&", next_page_token), "timeframe=1Min");
                let next_page_bars = &mut StockFrame::grab_entire_json(ticker, next_page_uri);
                bars.append(next_page_bars);
                bars
            }
        };

        return bars;
    }
    fn grab_latest_data(start: OffsetDateTime, end: OffsetDateTime, tickers: &Vec<String>) -> DataFrame {
        let mut df = DataFrame::default();

        for ticker in tickers {
            let uri = format!("https://data.alpaca.markets/v2/stocks/{}/bars?{}{}{}", ticker, format!("start={}&", start.format(&Rfc3339).unwrap()), format!("end={}&", end.format(&Rfc3339).unwrap()), "timeframe=1Min");
            let json_tree = serde_json::to_string(&StockFrame::grab_entire_json(ticker, &uri)).unwrap();
            let cursor = Cursor::new(json_tree);

            let mut tmp_df = JsonReader::new(cursor).finish().unwrap().clone().lazy().with_columns(
                [
                    polars::prelude::lit(ticker.as_str()).alias("symbol")
                ]
            ).collect().unwrap();

            tmp_df.set_column_names(vec!["close", "high", "low", "trade_count", "open", "timestamp", "volume", "vwap", "symbol"].as_slice()).expect("Collumn number mismatch");
            df = df.vstack(&tmp_df).unwrap();
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

        let end = OffsetDateTime::now_utc().replace_hour(0).unwrap().replace_minute(0).unwrap().replace_second(0).unwrap().replace_millisecond(0).unwrap().replace_microsecond(0).unwrap().replace_nanosecond(0).unwrap();
        let start = end.sub(Duration::days(30));

        let dataframe = StockFrame::grab_latest_data(start, end, &tickers_list).lazy().with_columns(
            columns_list[9..].iter().map(|s| polars::prelude::lit(polars::prelude::NULL).alias(s)).collect::<Vec<_>>().as_slice()
        ).collect().unwrap().select(&columns_list).unwrap().to_owned();

        StockFrame {
            columns: columns_list,
            tickers: tickers_list,
            frame: dataframe,
        }
    }
}