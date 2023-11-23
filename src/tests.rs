#[cfg(test)]
mod tests {
    use crate::environment::{Environment, Terminate};
    use crate::halfcheetahenv::HalfCheetahEnv;
    use crate::stockenv::StockEnv;
    use crate::stockframe::StockFrame;

    use dotenv::dotenv;
    use polars::export::chrono::{Duration, Utc};
    use polars::prelude::FillNullStrategy;
    use rand::prelude::{Distribution, StdRng};
    use rand::SeedableRng;

    #[test]
    fn test_halfcheetah_env() {
        let mut env = HalfCheetahEnv::new(None, None, None, None, None, None, None);

        let mut rng = StdRng::from_entropy();
        let uniform = rand::distributions::Uniform::from(0f64..1f64);

        let mut iter = 0;
        while iter < 5 {
            let ts = env.step(
                (0..env.action_spec().shape)
                    .map(|_| uniform.sample(&mut rng))
                    .collect(),
            );

            println!(
                "step: {}, obs: {:?}, reward: {:?}",
                env.step,
                ts.observation(),
                ts.reward()
            );

            if ts.as_ref().as_any().downcast_ref::<Terminate>().is_some() {
                println!("episode ended: {}", iter);
                iter += 1
            }
        }
    }

    #[test]
    #[ignore]
    fn test_stockenv() {
        dotenv().ok();

        let end = Utc::now()
            .date_naive()
            .and_hms_micro_opt(0, 0, 0, 0)
            .unwrap();
        let start = end - Duration::days(15);

        let _ = StockEnv::new(start, end);
    }
}
