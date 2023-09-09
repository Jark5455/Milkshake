use std::ffi::c_void;
use std::mem::{size_of, size_of_val};
use cuda_runtime_sys::cudaMalloc;
use cudnn::{cudaMemoryPtr, cudnnCreateTensorDescriptor, cudnnSetTensor4dDescriptor, TensorDescriptor};
use cudnn::utils::DataType;
use libc::PT_NULL;
use polars::export::num::Num;

// Stolen and reinterpreted from cudnn mnist sample
enum DeviceType {
    host,
    cuda
}

struct Blob<T> {
    is_tensor: bool,
    tensor_desc: Option<TensorDescriptor>,
    d_ptr: Option<cudaMemoryPtr>,

    h_ptr: Vec<T>,
    n: u32,
    c: u32,
    h: u32,
    w: u32
}

impl<T: Num> Blob<T> {
    pub(crate) fn new(n: Option<u32>, c: Option<u32>, h: Option<u32>, w: Option<u32>) -> Blob<T> {

        let dim_n = n.unwrap_or(1);
        let dim_c = c.unwrap_or(1);
        let dim_h = h.unwrap_or(1);
        let dim_w = w.unwrap_or(1);

        let mut h_vec = Vec::<T>::new();
        h_vec.resize_with((dim_n * dim_c * dim_h * dim_w) as usize, || { T::zero() });

        Blob {
            is_tensor: false,
            tensor_desc: None,
            d_ptr: None,
            h_ptr: h_vec,
            n: dim_n,
            c: dim_c,
            h: dim_h,
            w: dim_w
        }
    }

    pub(crate) fn new_1(size: &[u32; 4]) -> Blob<T> {

        let dim_n = size[0];
        let dim_c = size[0];
        let dim_h = size[0];
        let dim_w = size[0];

        Blob::new(Some(dim_n), Some(dim_c), Some(dim_h), Some(dim_w))
    }

    pub(crate) fn reset(&mut self, n: Option<u32>, c: Option<u32>, h: Option<u32>, w: Option<u32>) {
        self.n = n.unwrap_or(1);
        self.c = c.unwrap_or(1);
        self.h = h.unwrap_or(1);
        self.w = w.unwrap_or(1);

        let mut h_vec = Vec::<T>::new();
        h_vec.resize_with((self.n * self.c * self.h * self.w) as usize, || { T::zero() });

        self.h_ptr = h_vec;
        self.d_ptr = None;
        self.tensor_desc = None;
    }

    pub(crate) fn reset_1(&mut self, size: &[u32; 4]) {
        let dim_n = size[0];
        let dim_c = size[0];
        let dim_h = size[0];
        let dim_w = size[0];

        self.reset(Some(dim_n), Some(dim_c), Some(dim_h), Some(dim_w));
    }

    pub(crate) fn initCuda(&mut self) {
        if self.d_ptr.is_none() {
            unsafe {
                self.d_ptr = Some(PT_NULL as cudaMemoryPtr);
                cudaMalloc(self.d_ptr.as_mut().unwrap(), size_of::<T>() * self.h * self.c * self.h * self.w);
            }
        }

        self.d_ptr;
    }

    pub(crate) fn initTensor(&mut self) -> TensorDescriptor {
        if self.is_tensor {
            return self.tensor_desc.unwrap();
        }

        todo!();
    }
}