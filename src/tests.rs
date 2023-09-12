#[cfg(test)]
mod tests {
    use cust::CudaFlags;
    use cust::prelude::{Context, Device};
    use dotenv::dotenv;
    use polars::export::chrono::{Duration, Utc};
    use crate::cudnn_network::blob::{Blob, DeviceType};
    use crate::cudnn_network::loss::RegressionLoss;
    use crate::{context_s, device_s};
    use crate::stockenv::StockEnv;
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    #[ignore]
    fn test_regression() {
        cust::init(CudaFlags::empty()).expect("Failed to initialize cuda");

        device_s.with(|device_ref| {
            device_ref.replace(Some(Device::get_device(0).expect("Failed to find cuda device")));

            context_s.with(|context_ref| {
                context_ref.replace(Some(Context::new(device_ref.borrow().unwrap()).expect("Failed to create cuda context")))
            });
        });

        let mut target = Blob::<f32>::new(Some(5), Some(1), Some(1), Some(1));
        let mut predict = Blob::<f32>::new(Some(5), Some(1), Some(1), Some(1));

        target.h_ptr[0] = 1f32;
        target.h_ptr[1] = 1f32;
        target.h_ptr[2] = 0.2f32;
        target.h_ptr[3] = 0.5f32;
        target.h_ptr[4] = 1f32;

        predict.h_ptr[0] = 0f32;
        predict.h_ptr[1] = 0f32;
        predict.h_ptr[2] = 0f32;
        predict.h_ptr[3] = 0f32;
        predict.h_ptr[4] = 0f32;

        target.init_cuda();
        predict.init_cuda();

        target.to(DeviceType::cuda);
        predict.to(DeviceType::cuda);

        let mut regloss = RegressionLoss::new();
        regloss.init_workspace(5);

        println!("{}", regloss.loss(&mut predict, &mut target))
    }

    #[test]
    fn test_env() {
        dotenv().ok();

        let end = Utc::now().date_naive().and_hms_micro_opt(0, 0, 0, 0).unwrap();
        let start = end - Duration::days(15);

        let env = StockEnv::new(start, end);
    }

    #[test]
    #[ignore]
    fn test_file_io() {
        cust::init(CudaFlags::empty()).expect("Failed to initialize cuda");

        device_s.with(|device_ref| {
            device_ref.replace(Some(Device::get_device(0).expect("Failed to find cuda device")));

            context_s.with(|context_ref| {
                context_ref.replace(Some(Context::new(device_ref.borrow().unwrap()).expect("Failed to create cuda context")))
            });
        });

        let mut target = Blob::<f32>::new(Some(5), Some(1), Some(1), Some(1));
        target.init_cuda();

        target.h_ptr[0] = 1f32;
        target.h_ptr[1] = 1f32;
        target.h_ptr[2] = 0.2f32;
        target.h_ptr[3] = 0.5f32;
        target.h_ptr[4] = 1f32;

        target.to(DeviceType::cuda);
        target.file_write("target.bin".to_string());

        let mut pred = Blob::<f32>::new(Some(5), Some(1), Some(1), Some(1));
        pred.init_cuda();

        pred.file_read("target.bin".to_string());
        pred.to(DeviceType::cuda);

        assert_eq!(target.h_ptr, pred.h_ptr);
    }
}