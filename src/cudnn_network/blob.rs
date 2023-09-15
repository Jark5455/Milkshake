use anyhow::Result;
use std::ffi::c_void;
use std::fmt::{Display};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
use cudarc::cudnn::TensorDescriptor;
use cudarc::driver::{CudaSlice, DevicePtr};
use polars::export::num::Num;
use crate::{cudnn, device};

// The following code is stolen and reinterpreted from cudnn mnist sample
pub(crate) enum DeviceType {
    host,
    cuda
}

pub(crate) struct Blob<T: Num + Display> {
    pub tensor_desc: Option<TensorDescriptor<T>>,
    pub device_slice: Option<CudaSlice<T>>,
    pub host_slice: Vec<T>,

    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize
}

impl<T: Num + Display> Blob<T> {
    pub(crate) fn new(n: Option<usize>, c: Option<usize>, h: Option<usize>, w: Option<usize>) -> Blob<T> {

        let dim_n = n.unwrap_or(1);
        let dim_c = c.unwrap_or(1);
        let dim_h = h.unwrap_or(1);
        let dim_w = w.unwrap_or(1);

        let slice = vec![T::zero(); dim_n * dim_c * dim_h * dim_w];

        Blob {
            tensor_desc: None,
            device_slice: None,
            host_slice: slice,

            n: dim_n,
            c: dim_c,
            h: dim_h,
            w: dim_w
        }
    }

    pub(crate) fn new_1(size: &[usize; 4]) -> Blob<T> {
        let dim_n = size[0];
        let dim_c = size[0];
        let dim_h = size[0];
        let dim_w = size[0];

        Blob::new(Some(dim_n), Some(dim_c), Some(dim_h), Some(dim_w))
    }

    pub(crate) fn reset(&mut self, n: Option<usize>, c: Option<usize>, h: Option<usize>, w: Option<usize>) {
        self.n = n.unwrap_or(1);
        self.c = c.unwrap_or(1);
        self.h = h.unwrap_or(1);
        self.w = w.unwrap_or(1);

        self.host_slice = vec![T::zero(); self.n * self.c * self.h * self.w];

        self.device_slice = None;
        self.tensor_desc = None;
    }

    pub(crate) fn reset_1(&mut self, size: &[usize; 4]) {
        let dim_n = size[0];
        let dim_c = size[0];
        let dim_h = size[0];
        let dim_w = size[0];

        self.reset(Some(dim_n), Some(dim_c), Some(dim_h), Some(dim_w));
    }

    pub(crate) fn cuda(&mut self) -> &mut CudaSlice<T> {
        if self.device_slice.is_none() {
            self.device_slice = Some(device.alloc_zeros(self.n * self.c * self.h * self.w).expect("Failed to allocate CUDA slice"))
        }

        self.device_slice.as_mut().unwrap()
    }

    pub(crate) fn init_tensor(&mut self) -> &mut TensorDescriptor<T> {
        if self.tensor_desc.is_none() {

            let n = self.n as i32;
            let c = self.c as i32;
            let h = self.h as i32;
            let w = self.w as i32;

            self.tensor_desc = Some(cudnn.create_4d_tensor(CUDNN_TENSOR_NCHW, [n, c, h, w]).expect("Failed to create tensor descriptor"));
        }

        self.tensor_desc.as_mut().unwrap()
    }

    pub(crate) fn to(&mut self, target: DeviceType) {
        match target {
            DeviceType::host => {
                self.host_slice = device.dtoh_sync_copy(self.cuda()).expect("Failed to copy from device to host");
            }

            DeviceType::cuda => {
                self.device_slice = Some(device.htod_sync_copy(self.host_slice.as_slice()).expect("Failed to copy from host to device"));
            }
        }
    }

    pub(crate) fn print(&mut self, name: String, view_param: bool, batch: Option<u32>, width: Option<u32>) {
        let num_batch = batch.unwrap_or(1);
        let width = width.unwrap_or(16);

        print!("**{}\t: ({})\t", name, self.c * self.h * self.w);
        print!(".n: {}, .c: {}, .h: {}, .w: {}", self.n, self.c, self.h, self.w);
        println!("\t(h: {:p}, d: {:p})", self.host_slice.as_ptr(), self.device_slice.unwrap().device_ptr() as *mut c_void);

        if view_param {
            self.to(DeviceType::host);

            for n in 0..num_batch {
                if num_batch > 1 {
                    println!("<--- batch[{}] --->", {n});
                }

                let mut count = 0;
                while count < self.c * self.h * self.w {
                    print!("\t");

                    let mut s = 0;
                    while s < width && count < self.c * self.h * self.w {
                        print!("{:}\t", self.device_slice[(self.c * self.h * self.w) * n as usize + count]);
                        count += 1;
                        s += 1;
                    }

                    println!();
                }
            }
        }
    }

    pub(crate) fn file_read(&mut self, name: String) -> Result<()> {
        let input = File::open(name)?;
        let buf = BufReader::new(input);

        let data = buf.bytes().collect::<Result<Vec<_>, _> >()?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.host_slice.as_mut_ptr().cast::<u8>(), data.len()); }

        self.to(DeviceType::cuda);
        Ok(())
    }

    pub(crate) fn file_write(&mut self, name: String) -> Result<()> {
        self.to(DeviceType::host);
        let mut out = File::create(name)?;

        unsafe {
            let mut clone = self.host_slice.clone();
            let stride = std::mem::size_of::<T>() / std::mem::size_of::<u8>();
            let str = String::from_raw_parts(clone.as_mut_ptr().cast::<u8>(), clone.len() * stride, clone.capacity() * stride);
            write!(out, "{}", str)?;

            std::mem::forget(clone);
        }

        Ok(())
    }
}