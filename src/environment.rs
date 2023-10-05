use std::any::{Any, TypeId};

pub struct Spec {
    pub min: f64,
    pub max: f64,

    // no multidimensional spec support yet (just flatten the action space)
    pub shape: u32,
}

pub trait Trajectory {
    fn observation(&self) -> Option<Vec<f64>>;
    fn reward(&self) -> Option<f64>;
    fn as_any(&self) -> &dyn Any;
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
    fn step(&mut self, _action: Vec<f64>) -> Box<dyn Trajectory>;
}

impl Trajectory for Transition {
    fn observation(&self) -> Option<Vec<f64>> {
        Some(self.observation.clone())
    }
    fn reward(&self) -> Option<f64> {
        Some(self.reward)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Trajectory for Terminate {
    fn observation(&self) -> Option<Vec<f64>> {
        Some(self.observation.clone())
    }
    fn reward(&self) -> Option<f64> {
        return Some(self.reward);
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Trajectory for Restart {
    fn observation(&self) -> Option<Vec<f64>> {
        Some(self.observation.clone())
    }
    fn reward(&self) -> Option<f64> {
        return None;
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
