use anyhow::Result;
use std::ffi::c_void;
use std::fmt::{Display};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use cust::memory::bytemuck::Zeroable;
use cust::memory::DeviceMemory;
use rcudnn::{utils::DataType, TensorDescriptor};
use cust::prelude::{CopyDestination, DeviceBuffer, DeviceCopyExt};
use polars::export::num::Num;

// The following code is stolen and reinterpreted from cudnn mnist sample
pub(crate) enum DeviceType {
    host,
    cuda
}

pub(crate) struct Blob<T: Num + DeviceCopyExt + Display> {
    pub tensor_desc: Option<TensorDescriptor>,
    pub d_ptr: Option<DeviceBuffer<T>>,

    pub h_ptr: Vec<T>,
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize
}

impl<T: Num + DeviceCopyExt + Display + Zeroable> Blob<T> {
    pub(crate) fn new(n: Option<usize>, c: Option<usize>, h: Option<usize>, w: Option<usize>) -> Blob<T> {

        let dim_n = n.unwrap_or(1);
        let dim_c = c.unwrap_or(1);
        let dim_h = h.unwrap_or(1);
        let dim_w = w.unwrap_or(1);

        let mut h_vec = Vec::<T>::new();
        h_vec.resize_with(dim_n * dim_c * dim_h * dim_w, || { T::zero() });

        Blob {
            tensor_desc: None,
            d_ptr: None,
            h_ptr: h_vec,
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

        let mut h_vec = Vec::<T>::new();
        h_vec.resize_with(self.n * self.c * self.h * self.w, || { T::zero() });

        self.h_ptr = h_vec;
        self.d_ptr = None;
        self.tensor_desc = None;
    }

    pub(crate) fn reset_1(&mut self, size: &[usize; 4]) {
        let dim_n = size[0];
        let dim_c = size[0];
        let dim_h = size[0];
        let dim_w = size[0];

        self.reset(Some(dim_n), Some(dim_c), Some(dim_h), Some(dim_w));
    }

    pub(crate) fn init_cuda(&mut self) -> &mut DeviceBuffer<T> {
        if self.d_ptr.is_none() {
            self.d_ptr = Some(DeviceBuffer::zeroed(self.n * self.c * self.h * self.w).expect("Failed to allocate CUDA memory"))
        }

        self.d_ptr.as_mut().unwrap()
    }

    pub(crate) fn init_tensor(&mut self) -> &mut TensorDescriptor {
        if self.tensor_desc.is_none() {

            let n = self.n as i32;
            let c = self.c as i32;
            let h = self.h as i32;
            let w = self.w as i32;

            // I dont understand stride and mem alignment too much but I think this is right
            self.tensor_desc = Some(TensorDescriptor::new(&[n, c, h, w], &[c * h * w, h * w, w, 1], DataType::Float).expect("Failed to create tensor descriptor"));
        }

        self.tensor_desc.as_mut().unwrap()
    }

    pub(crate) fn to(&mut self, target: DeviceType) {
        match target {
            DeviceType::host => {
                self.d_ptr.as_mut().unwrap().copy_to(self.h_ptr.as_mut_slice()).expect("Failed to copy from device to host");
            }

            DeviceType::cuda => {
                self.d_ptr.as_mut().unwrap().copy_from(self.h_ptr.as_slice()).expect("Failed to copy from host to device");
            }
        }
    }

    pub(crate) fn print(&mut self, name: String, view_param: bool, batch: Option<u32>, width: Option<u32>) {
        let num_batch = batch.unwrap_or(1);
        let width = width.unwrap_or(16);

        print!("**{}\t: ({})\t", name, self.c * self.h * self.w);
        print!(".n: {}, .c: {}, .h: {}, .w: {}", self.n, self.c, self.h, self.w);
        println!("\t(h: {:p}, d: {:p})", self.h_ptr.as_ptr(), self.d_ptr.as_ref().unwrap().as_raw_ptr() as *mut c_void);

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
                        print!("{:}\t", self.h_ptr[(self.c * self.h * self.w) * n as usize + count]);
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
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.h_ptr.as_mut_ptr().cast::<u8>(), data.len()); }

        self.to(DeviceType::cuda);

        Ok(())
    }

    pub(crate) fn file_write(&mut self, name: String) -> Result<()> {
        self.to(DeviceType::host);
        let mut out = File::create(name)?;

        unsafe {
            let mut clone = self.h_ptr.clone();
            let stride = std::mem::size_of::<T>() / std::mem::size_of::<u8>();
            let str = String::from_raw_parts(clone.as_mut_ptr().cast::<u8>(), clone.len() * stride, clone.capacity() * stride);
            write!(out, "{}", str)?;

            std::mem::forget(clone);
        }

        Ok(())
    }
}