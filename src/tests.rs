#[cfg(test)]
mod tests {
    use std::any::{Any, TypeId};
    use std::ops::Deref;
    use crate::stockenv::StockEnv;
    use crate::stockframe::StockFrame;
    use crate::halfcheetahenv::HalfCheetahEnv;
    use dotenv::dotenv;
    use polars::export::chrono::{Duration, Utc};
    use polars::prelude::FillNullStrategy;
    use rand::prelude::{Distribution, StdRng};
    use rand::SeedableRng;
    use crate::environment::{Environment, Terminate};

    #[test]
    #[ignore]
    fn test_halfcheetah_env() {
        let mut env = HalfCheetahEnv::new(None, None, None, None, None, None, None);
        let mut rng = StdRng::from_entropy();
        let uniform = rand::distributions::Uniform::from(0f64..1f64);

        let mut iter = 0;
        while iter < 5 {
            let ts = env.step((0..env.action_spec().shape).map(|idx| uniform.sample(&mut rng)).collect());

            println!("step: {}, obs: {:?}, reward: {:?}", env.step, ts.observation(), ts.reward());

            if ts.as_ref().as_any().downcast_ref::<Terminate>().is_some() {
                println!("episode ended: {}", iter);
                iter += 1
            }
        }
    }

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
