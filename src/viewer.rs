// SHOULD BE STATIC, ONLY 1 INSTANCE

extern crate glfw_bindgen;

use glfw_bindgen::{
    glfwCreateWindow, glfwDestroyWindow, glfwGetFramebufferSize, glfwGetPrimaryMonitor,
    glfwGetVideoMode, glfwGetWindowSize, glfwInit, glfwMakeContextCurrent, glfwPollEvents,
    glfwSetKeyCallback, glfwSwapBuffers, glfwSwapInterval, glfwTerminate, glfwWindowShouldClose,
    GLFWwindow, GLFW_KEY_ESCAPE, GLFW_TRUE,
};
use libc::{c_char, c_int};

use std::mem::MaybeUninit;
use std::ptr::{copy_nonoverlapping, null_mut};

use crate::environment::MujocoEnvironment;
use crate::td3::TD3;

pub struct Viewer<'vw> {
    window: &'vw mut GLFWwindow,
    scale: f64,

    cam: crate::mujoco::mjvCamera,
    opt: crate::mujoco::mjvOption,
    scene: crate::mujoco::mjvScene,
    context: crate::mujoco::mjrContext,

    env: Box<dyn MujocoEnvironment>,
    td3: TD3,
}

impl Viewer<'_> {
    pub fn new(
        env: Box<dyn MujocoEnvironment>,
        td3: TD3,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Self {
        unsafe {
            assert_eq!(glfwInit(), GLFW_TRUE as i32);

            let width =
                width.unwrap_or((*glfwGetVideoMode(glfwGetPrimaryMonitor())).width as u32 / 2);
            let height =
                height.unwrap_or((*glfwGetVideoMode(glfwGetPrimaryMonitor())).height as u32 / 2);

            let window_raw = glfwCreateWindow(
                width as c_int,
                height as c_int,
                "Milkshake".as_ptr() as *const c_char,
                null_mut(),
                null_mut(),
            );
            let window = Box::leak(Box::from_raw(window_raw));

            glfwMakeContextCurrent(window);
            glfwSwapInterval(1);

            let mut framebuffer_width = 0;
            let mut framebuffer_height = 0;

            glfwGetFramebufferSize(window, &mut framebuffer_width, &mut framebuffer_height);

            let mut window_height = 0;
            let mut window_width = 0;

            glfwGetWindowSize(window, &mut window_width, &mut window_height);

            let scale = framebuffer_width as f64 * (1f64 / window_width as f64);

            glfwSetKeyCallback(window, Some(Self::key_callback));

            let mut cam_uninit = MaybeUninit::uninit();
            let mut opt_uninit = MaybeUninit::uninit();
            let mut scene_uninit = MaybeUninit::uninit();
            let mut context_uninit = MaybeUninit::uninit();

            crate::mujoco::mjv_defaultCamera(cam_uninit.as_mut_ptr());
            crate::mujoco::mjv_defaultOption(opt_uninit.as_mut_ptr());
            crate::mujoco::mjv_defaultScene(scene_uninit.as_mut_ptr());
            crate::mujoco::mjr_defaultContext(context_uninit.as_mut_ptr());

            let cam = cam_uninit.assume_init();
            let opt = opt_uninit.assume_init();
            let scene = scene_uninit.assume_init();
            let context = context_uninit.assume_init();

            Viewer {
                window,
                scale,
                cam,
                opt,
                scene,
                context,
                env,
                td3,
            }
        }
    }

    pub fn render(&mut self) {
        unsafe {
            self.cam.type_ = crate::mujoco::mjtCamera__mjCAMERA_TRACKING as c_int;
            self.cam.trackbodyid = *self.env.model().cam_bodyid;

            crate::mujoco::mjv_makeScene(self.env.model(), &mut self.scene, 1000);
            crate::mujoco::mjr_makeContext(
                self.env.model(),
                &mut self.context,
                crate::mujoco::mjtFontScale__mjFONTSCALE_100 as c_int,
            );
        };

        while unsafe { glfwWindowShouldClose(self.window) == 0 } {
            let obs = self.env.observation();
            let action = self.td3.select_action(obs);
            unsafe {
                copy_nonoverlapping(action.as_ptr(), self.env.data().ctrl, action.len());
            }

            let refreshrate =
                unsafe { (*glfwGetVideoMode(glfwGetPrimaryMonitor())).refreshRate as f64 };
            let simstart = self.env.data().time;
            while self.env.data().time - simstart < 1f64 / refreshrate {
                println!("{}", self.env.data().time);
                unsafe { crate::mujoco::mj_step(self.env.model(), self.env.data()) };
            }

            let mut viewport = crate::mujoco::mjrRect {
                left: 0,
                bottom: 0,
                width: 0,
                height: 0,
            };

            unsafe {
                glfwGetFramebufferSize(self.window, &mut viewport.width, &mut viewport.height);
                crate::mujoco::mjv_updateScene(
                    self.env.model(),
                    self.env.data(),
                    &self.opt,
                    null_mut(),
                    &mut self.cam,
                    crate::mujoco::mjtCatBit__mjCAT_ALL as c_int,
                    &mut self.scene,
                );
                crate::mujoco::mjv_updateCamera(
                    self.env.model(),
                    self.env.data(),
                    &mut self.cam,
                    &mut self.scene,
                );
                crate::mujoco::mjr_render(viewport, &mut self.scene, &self.context);
                glfwSwapBuffers(self.window);
                glfwPollEvents();
            };
        }

        unsafe { crate::mujoco::mjv_freeScene(&mut self.scene) };
    }

    unsafe extern "C" fn key_callback(
        _window: *mut GLFWwindow,
        key: c_int,
        _scancode: c_int,
        _action: c_int,
        _mods: c_int,
    ) {
        if key == GLFW_KEY_ESCAPE as i32 {
            std::process::exit(0);
        }
    }
}

impl Drop for Viewer<'_> {
    fn drop(&mut self) {
        unsafe {
            glfwDestroyWindow(self.window);
            glfwTerminate();

            crate::mujoco::mjr_freeContext(&mut self.context);
        }
    }
}
