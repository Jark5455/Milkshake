// SHOULD BE STATIC, ONLY 1 INSTANCE

use std::ptr::{null_mut};
use glfw_bindgen::{GLFW_TRUE, glfwCreateWindow, glfwDestroyWindow, glfwGetPrimaryMonitor, glfwGetVideoMode, glfwInit, glfwMakeContextCurrent, glfwSwapInterval, glfwTerminate, GLFWwindow};
use libc::{c_char, c_int};

struct Viewer {
    window: &'static mut GLFWwindow
}

impl Viewer {
    pub fn new(width: Option<u32>, height: Option<u32>) -> Self {
        unsafe {
            assert_eq!(glfwInit(), GLFW_TRUE as i32);

            let width = unsafe { width.unwrap_or((*glfwGetVideoMode(glfwGetPrimaryMonitor())).width as u32 / 2) };
            let height = unsafe { height.unwrap_or((*glfwGetVideoMode(glfwGetPrimaryMonitor())).height as u32 / 2) };

            let window_raw = glfwCreateWindow(width as c_int, height as c_int, "Milkshake".as_ptr() as *const c_char, null_mut(), null_mut());
            let window = Box::leak(Box::from_raw(window_raw));

            glfwMakeContextCurrent(window);
            glfwSwapInterval(1);

            Viewer { window }
        }
    }
}

impl Drop for Viewer {
    fn drop(&mut self) {
        unsafe {
            glfwDestroyWindow(self.window);
            glfwTerminate();
        }
    }
}