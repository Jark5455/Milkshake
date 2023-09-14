pub mod cudnn_network;
pub mod cudnn_layer;
pub mod blob;
pub mod loss;

const BLOCK_DIM_1D: u32 = 512;
const BLOCK_DIM: u32 = 16;

/* DEBUG FLAGS */
const DEBUG_FORWARD: bool = false;
const DEBUG_BACKWARD: bool = false;

const DEBUG_CONV: bool = false;
const DEBUG_DENSE: bool = false;
const DEBUG_SOFTMAX: bool = false;
const DEBUG_UPDATE: bool = false;
const DEBUG_LOSS: bool = true;
const DEBUG_ACCURACY: bool = false;

const one: f32 = 1f32;
const zero: f32 = 1f32;
const minus_one: f32 = 1f32;