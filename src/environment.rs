pub(crate) struct Spec {
    pub min: f64,
    pub max: f64,

    // no multidimensional spec support yet
    pub shape: u32
}

pub(crate) trait Trajectory {
    fn reward(&self) -> f64 {
        return 0.0;
    }
}

pub(crate) struct Transition {
    pub reward: f64
}
pub(crate) struct Terminate {
    pub reward: f64
}
pub(crate) struct Restart {
    pub reward: f64
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

    fn step(&self) -> Box<dyn Trajectory> {
        return Box::new(Transition{ reward: 0.0 });
    }

    fn reset(&self) -> Box<dyn Trajectory> {
        return Box::new(Restart{ reward: 0.0 });
    }
}

impl Trajectory for Transition {
    fn reward(&self) -> f64 {
        return self.reward;
    }
}

impl Trajectory for Terminate {
    fn reward(&self) -> f64 {
        return self.reward;
    }
}

impl Trajectory for Restart {
    fn reward(&self) -> f64 {
        return self.reward;
    }
}