pub(crate) struct Spec {
    pub min: f64,
    pub max: f64,

    // no multidimensional spec support yet (just flatten the action space)
    pub shape: u32
}

pub(crate) trait Trajectory {
    fn observation(&self) -> Option<Vec<f64>> {
        return Some(vec![]);
    }
    fn reward(&self) -> Option<f64> {
        return Some(0.0);
    }
}

pub(crate) struct Transition {
    pub observation: Vec<f64>,
    pub reward: f64
}
pub(crate) struct Terminate {
    pub observation: Vec<f64>,
    pub reward: f64
}
pub(crate) struct Restart {
    pub observation: Vec<f64>
}

pub(crate) trait Environment {
    fn action_spec (&self) -> Spec {
        return Spec{
            min: 0.0,
            max: 0.0,
            shape: 0,
        };
    }

    fn observation_spec(&self) -> Spec {
        return Spec{
            min: 0.0,
            max: 0.0,
            shape: 0,
        };
    }

    fn step(&mut self, action: Vec<f64>) -> Box<dyn Trajectory> {
        return Box::new(Transition{ observation: vec![], reward: 0.0 });
    }
}

impl Trajectory for Transition {
    fn observation(&self) -> Option<Vec<f64>> { Some(self.observation.clone()) }
    fn reward(&self) -> Option<f64> { Some(self.reward) }
}

impl Trajectory for Terminate {
    fn observation(&self) -> Option<Vec<f64>> { Some(self.observation.clone()) }
    fn reward(&self) -> f64 {
        return self.reward;
    }
}

impl Trajectory for Restart {
    fn observation(&self) -> Option<Vec<f64>> { Some(self.observation.clone()) }
    fn reward(&self) -> Option<f64> { return None }
}