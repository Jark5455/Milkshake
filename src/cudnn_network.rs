use std::cell::RefCell;
use cudnn::Cudnn;

struct Network {
    cudnnHandle: Cudnn,

}

impl Network {
    pub(crate) fn new() {
        let cudnn_handle = Cudnn::new().unwrap();
    }
}