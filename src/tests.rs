#[cfg(test)]
mod tests {
    use crate::stockenv::StockEnv;
    use crate::stockframe::StockFrame;
    use dotenv::dotenv;
    use polars::export::chrono::{Duration, Utc};
    use polars::prelude::FillNullStrategy;
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    #[ignore]
    fn test_env() {
        dotenv().ok();

        let end = Utc::now()
            .date_naive()
            .and_hms_micro_opt(0, 0, 0, 0)
            .unwrap();
        let start = end - Duration::days(15);

        let env = StockEnv::new(start, end);
    }

    #[test]
    #[ignore]
    fn test_stockframe() {
        dotenv().ok();

        let end = Utc::now()
            .date_naive()
            .and_hms_micro_opt(0, 0, 0, 0)
            .unwrap();

        let start = end - Duration::days(15);

        let mut stockframe = StockFrame::new(
            Some(
                vec!["AAPL", "TLSA"]
                    .iter()
                    .map(|s| String::from(*s))
                    .collect(),
            ),
            Some(start.clone()),
            Some(end.clone()),
        );

        stockframe.parse_dt_column();
        stockframe.fill_date_range();
        stockframe.fill_nulls();

        unsafe {
            stockframe.calc_technical_indicators();
        }

        // fill volume, vwap, and trade_count with zeros
        stockframe.frame = Box::new(
            stockframe
                .clone()
                .frame
                .fill_null(FillNullStrategy::Zero)
                .unwrap(),
        );
        stockframe.clean();

        // sort
        stockframe.update_symbol_groups();
        stockframe.frame = Box::new(
            stockframe
                .clone()
                .frame
                .sort(&["symbol", "timestamp"], vec![false, false], false)
                .unwrap(),
        );

        println!("{}", stockframe.frame);
    }
}
