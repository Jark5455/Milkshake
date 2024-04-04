// SHOULD BE STATIC, ONLY 1 INSTANCE

extern crate glfw_bindgen;

use crate::environment::Mujoco;
use crate::td3::TD3;

pub struct Viewer<'vw> {
    window: &'vw mut glfw_bindgen::GLFWwindow,
    scale: f64,

    cam: crate::wrappers::mujoco::mjvCamera,
    opt: crate::wrappers::mujoco::mjvOption,
    scene: crate::wrappers::mujoco::mjvScene,
    context: crate::wrappers::mujoco::mjrContext,

    env: Box<dyn Mujoco>,
    td3: TD3,
}

impl Viewer<'_> {
    pub fn new(
        env: Box<dyn Mujoco>,
        td3: TD3,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Self {
        unsafe {
            assert_eq!(glfw_bindgen::glfwInit(), glfw_bindgen::GLFW_TRUE as i32);

            let width = width.unwrap_or(
                (*glfw_bindgen::glfwGetVideoMode(glfw_bindgen::glfwGetPrimaryMonitor())).width
                    as u32
                    / 2,
            );
            let height = height.unwrap_or(
                (*glfw_bindgen::glfwGetVideoMode(glfw_bindgen::glfwGetPrimaryMonitor())).height
                    as u32
                    / 2,
            );

            let window_raw = glfw_bindgen::glfwCreateWindow(
                width as libc::c_int,
                height as libc::c_int,
                "Milkshake".as_ptr() as *const libc::c_char,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            let window = Box::leak(Box::from_raw(window_raw));

            glfw_bindgen::glfwMakeContextCurrent(window);
            glfw_bindgen::glfwSwapInterval(1);

            let mut framebuffer_width = 0;
            let mut framebuffer_height = 0;

            glfw_bindgen::glfwGetFramebufferSize(
                window,
                &mut framebuffer_width,
                &mut framebuffer_height,
            );

            let mut window_height = 0;
            let mut window_width = 0;

            glfw_bindgen::glfwGetWindowSize(window, &mut window_width, &mut window_height);

            let scale = framebuffer_width as f64 * (1f64 / window_width as f64);

            glfw_bindgen::glfwSetKeyCallback(window, Some(Self::key_callback));

            let mut cam_uninit = std::mem::MaybeUninit::uninit();
            let mut opt_uninit = std::mem::MaybeUninit::uninit();
            let mut scene_uninit = std::mem::MaybeUninit::uninit();
            let mut context_uninit = std::mem::MaybeUninit::uninit();

            crate::wrappers::mujoco::mjv_defaultCamera(cam_uninit.as_mut_ptr());
            crate::wrappers::mujoco::mjv_defaultOption(opt_uninit.as_mut_ptr());
            crate::wrappers::mujoco::mjv_defaultScene(scene_uninit.as_mut_ptr());
            crate::wrappers::mujoco::mjr_defaultContext(context_uninit.as_mut_ptr());

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
            self.cam.type_ = crate::wrappers::mujoco::mjtCamera__mjCAMERA_TRACKING as libc::c_int;
            self.cam.trackbodyid = *self.env.model().cam_bodyid;

            crate::wrappers::mujoco::mjv_makeScene(self.env.model(), &mut self.scene, 1000);
            crate::wrappers::mujoco::mjr_makeContext(
                self.env.model(),
                &mut self.context,
                crate::wrappers::mujoco::mjtFontScale__mjFONTSCALE_100 as libc::c_int,
            );
        };

        let refreshrate = unsafe {
            (*glfw_bindgen::glfwGetVideoMode(glfw_bindgen::glfwGetPrimaryMonitor())).refreshRate
                as f64
        };

        while unsafe { glfw_bindgen::glfwWindowShouldClose(self.window) == 0 } {
            let obs = self.env.observation();
            let action = self.td3.select_action(obs);
            unsafe {
                std::ptr::copy_nonoverlapping(action.as_ptr(), self.env.data().ctrl, action.len());
            }

            let simstart = self.env.data().time;
            while self.env.data().time - simstart < 1f64 / refreshrate {
                println!("Time: {:.3}", self.env.data().time);
                unsafe { crate::wrappers::mujoco::mj_step(self.env.model(), self.env.data()) };
            }

            let mut viewport = crate::wrappers::mujoco::mjrRect {
                left: 0,
                bottom: 0,
                width: 0,
                height: 0,
            };

            unsafe {
                glfw_bindgen::glfwGetFramebufferSize(
                    self.window,
                    &mut viewport.width,
                    &mut viewport.height,
                );
                crate::wrappers::mujoco::mjv_updateScene(
                    self.env.model(),
                    self.env.data(),
                    &self.opt,
                    std::ptr::null_mut(),
                    &mut self.cam,
                    crate::wrappers::mujoco::mjtCatBit__mjCAT_ALL as libc::c_int,
                    &mut self.scene,
                );
                crate::wrappers::mujoco::mjv_updateCamera(
                    self.env.model(),
                    self.env.data(),
                    &mut self.cam,
                    &mut self.scene,
                );
                crate::wrappers::mujoco::mjr_render(viewport, &mut self.scene, &self.context);
                glfw_bindgen::glfwSwapBuffers(self.window);
                glfw_bindgen::glfwPollEvents();
            };
        }

        unsafe { crate::wrappers::mujoco::mjv_freeScene(&mut self.scene) };
    }

    unsafe extern "C" fn key_callback(
        _window: *mut glfw_bindgen::GLFWwindow,
        key: libc::c_int,
        _scancode: libc::c_int,
        _action: libc::c_int,
        _mods: libc::c_int,
    ) {
        if key == glfw_bindgen::GLFW_KEY_ESCAPE as i32 {
            std::process::exit(0);
        }
    }
}

impl Drop for Viewer<'_> {
    fn drop(&mut self) {
        unsafe {
            glfw_bindgen::glfwDestroyWindow(self.window);
            glfw_bindgen::glfwTerminate();

            crate::wrappers::mujoco::mjr_freeContext(&mut self.context);
        }
    }
}
