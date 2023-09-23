#[cfg(test)]
mod tests {
    use crate::stockenv::StockEnv;
    use dotenv::dotenv;
    use polars::export::chrono::{Duration, Utc};
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
}
