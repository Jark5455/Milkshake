pub mod cudnn_network;
pub mod cudnn_layer;
pub mod blob;
pub mod loss;

const BLOCK_DIM_1D: u32 = 512;
const BLOCK_DIM: u32 = 16;

/* DEBUG FLAGS */
const DEBUG_FORWARD: u32 = 0;
const DEBUG_BACKWARD: u32 = 0;

const DEBUG_CONV: u32 = 0;
const DEBUG_DENSE: u32 = 0;
const DEBUG_SOFTMAX: u32 = 0;
const DEBUG_UPDATE: u32 = 0;
const DEBUG_LOSS: u32 = 0;
const DEBUG_ACCURACY: u32 = 0;