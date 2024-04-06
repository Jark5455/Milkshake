pub mod stockenv;
pub mod halfcheetahenv;
pub mod antenv;

pub struct Spec {
    pub min: f64,
    pub max: f64,

    // no multidimensional spec support yet (just flatten the action space)
    pub shape: u32,
}

pub trait Trajectory {
    fn observation(&self) -> Vec<f64>;
    fn reward(&self) -> Option<f64>;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub struct Transition {
    pub observation: Vec<f64>,
    pub reward: f64,
}
pub struct Terminate {
    pub observation: Vec<f64>,
    pub reward: f64,
}
pub struct Restart {
    pub observation: Vec<f64>,
}

pub trait Environment {
    fn action_spec(&self) -> Spec;
    fn observation_spec(&self) -> Spec;
    fn step(&mut self, action: Vec<f64>) -> Box<dyn Trajectory>;
    fn reset(&mut self) -> Box<dyn Trajectory>;
}

pub trait Mujoco: Environment {
    fn model(&mut self) -> &mut crate::wrappers::mujoco::mjModel;

    fn data(&mut self) -> &mut crate::wrappers::mujoco::mjData;

    fn observation(&self) -> Vec<f64>;
}

impl Trajectory for Transition {
    fn observation(&self) -> Vec<f64> {
        self.observation.clone()
    }
    fn reward(&self) -> Option<f64> {
        Some(self.reward)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Trajectory for Terminate {
    fn observation(&self) -> Vec<f64> {
        self.observation.clone()
    }
    fn reward(&self) -> Option<f64> {
        Some(self.reward)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Trajectory for Restart {
    fn observation(&self) -> Vec<f64> {
        self.observation.clone()
    }
    fn reward(&self) -> Option<f64> {
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
