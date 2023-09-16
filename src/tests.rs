#[cfg(test)]
mod tests {
    use cudarc::nvrtc::compile_ptx;
    use dotenv::dotenv;
    use polars::export::chrono::{Duration, Utc};
    use crate::cudnn_network::blob::{Blob, DeviceType};
    use crate::cudnn_network::loss::RegressionLoss;
    use crate::{device, init_one_vec_kernel_code, mse_loss_kernel_code};
    use crate::stockenv::StockEnv;
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    #[ignore]
    fn test_regression() {

        let mse_loss_kernel = compile_ptx(mse_loss_kernel_code.clone()).expect("Failed to compile mse loss kernel");
        device.load_ptx(mse_loss_kernel, "mse_loss", &["mse_loss_kernel"]).expect("Failed to load mse loss kernel");

        let init_one_vec_kernel = compile_ptx(init_one_vec_kernel_code.clone()).expect("Failed to compile init one vec kernel");
        device.load_ptx(init_one_vec_kernel, "init_one_vec", &["init_one_vec"]).expect("Failed to load init one vec kernel");

        let mut target = Blob::<f32>::new(Some(5), Some(1), Some(1), Some(1));
        let mut predict = Blob::<f32>::new(Some(5), Some(1), Some(1), Some(1));

        target.host_slice[0] = 1f32;
        target.host_slice[1] = 1f32;
        target.host_slice[2] = 0.2f32;
        target.host_slice[3] = 0.5f32;
        target.host_slice[4] = 1f32;

        predict.host_slice[0] = 0f32;
        predict.host_slice[1] = 0f32;
        predict.host_slice[2] = 0f32;
        predict.host_slice[3] = 0f32;
        predict.host_slice[4] = 0f32;

        target.cuda();
        predict.cuda();

        target.to(DeviceType::cuda);
        predict.to(DeviceType::cuda);

        let mut regloss = RegressionLoss::new();
        regloss.init_workspace(5);

        println!("{}", regloss.loss(&mut predict, &mut target))
    }

    #[test]
    #[ignore]
    fn test_env() {
        dotenv().ok();

        let end = Utc::now().date_naive().and_hms_micro_opt(0, 0, 0, 0).unwrap();
        let start = end - Duration::days(15);

        let env = StockEnv::new(start, end);
    }

    #[test]
    #[ignore]
    fn test_file_io() {
        let mut target = Blob::<f32>::new(Some(5), Some(1), Some(1), Some(1));
        target.cuda();

        target.host_slice[0] = 1f32;
        target.host_slice[1] = 1f32;
        target.host_slice[2] = 0.2f32;
        target.host_slice[3] = 0.5f32;
        target.host_slice[4] = 1f32;

        target.to(DeviceType::cuda);
        target.file_write("target.banan".to_string()).expect("Failed to write to file");

        let mut pred = Blob::<f32>::new(Some(5), Some(1), Some(1), Some(1));
        pred.cuda();

        pred.file_read("target.banan".to_string()).expect("Failed to read from file");
        pred.to(DeviceType::cuda);

        assert_eq!(target.host_slice, pred.host_slice);
    }
}