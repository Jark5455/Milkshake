pub(crate) struct Spec {
    pub min: u32,
    pub max: u32,
    pub shape: ()
}

pub(crate) trait Trajectory {

}

pub(crate) struct Transition {

}
pub(crate) struct Terminate {

}
pub(crate) struct Restart {

}

pub(crate) trait Environment {
    fn action_spec () -> Spec {
        return Spec{
            min: 0,
            max: 0,
            shape: (),
        };
    }

    fn observation_spec() -> Spec {
        return Spec{
            min: 0,
            max: 0,
            shape: (),
        };
    }

    fn step() -> Box<dyn Trajectory> {
        return Box::new(Transition{});
    }

    fn reset() -> Box<dyn Trajectory> {
        return Box::new(Restart{});
    }
}