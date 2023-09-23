pub(crate) struct Spec {
    pub min: f64,
    pub max: f64,

    // no multidimensional spec support yet (just flatten the action space)
    pub shape: u32,
}

pub(crate) trait Trajectory {
    fn observation(&self) -> Option<Vec<f64>>;
    fn reward(&self) -> Option<f64>;
}

pub(crate) struct Transition {
    pub observation: Vec<f64>,
    pub reward: f64,
}
pub(crate) struct Terminate {
    pub observation: Vec<f64>,
    pub reward: f64,
}
pub(crate) struct Restart {
    pub observation: Vec<f64>,
}

pub(crate) trait Environment {
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
}

impl Trajectory for Terminate {
    fn observation(&self) -> Option<Vec<f64>> {
        Some(self.observation.clone())
    }
    fn reward(&self) -> Option<f64> {
        return Some(self.reward);
    }
}

impl Trajectory for Restart {
    fn observation(&self) -> Option<Vec<f64>> {
        Some(self.observation.clone())
    }
    fn reward(&self) -> Option<f64> {
        return None;
    }
}
