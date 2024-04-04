extern crate anyhow;
extern crate serde;
extern crate tch;

use crate::device;
use crate::replay_buffer::ReplayBuffer;

use crate::optimizer::adam::ADAM;
use crate::optimizer::cmaes::CMAES;
use crate::optimizer::MilkshakeOptimizer;

#[derive(Debug)]
pub struct MilkshakeLayer {
    pub layer: tch::nn::Linear,
    pub input: i64,
    pub output: i64,
}

impl tch::nn::Module for MilkshakeLayer {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        self.layer.forward(xs)
    }
}

impl serde::Serialize for MilkshakeLayer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut struct_serializer = serializer.serialize_struct("MilkshakeLayer", 2)?;

        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(
            &mut struct_serializer,
            "input_dim",
            &self.input
        )?;

        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(
            &mut struct_serializer,
            "output_dim",
            &self.output,
        )?;


        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::end(struct_serializer)
    }
}

#[derive(Debug)]
pub struct MilkshakeNetwork {
    pub layers: Vec<MilkshakeLayer>,
}

impl tch::nn::Module for MilkshakeNetwork {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        let mut alpha = self
            .layers
            .first()
            .unwrap()
            .forward(&xs.totype(tch::Kind::Float))
            .relu();

        for layer in &self.layers[1..1] {
            alpha = layer.forward(&alpha).relu();
        }

        self.layers.last().unwrap().forward(&alpha).tanh()
    }
}

impl serde::Serialize for MilkshakeNetwork {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq_serializer = serializer.serialize_seq(Some(self.layers.len()))?;

        for layer in &self.layers {
            <<S as serde::Serializer>::SerializeSeq as serde::ser::SerializeSeq>::serialize_element(&mut seq_serializer, layer)?;
        }

        <<S as serde::Serializer>::SerializeSeq as serde::ser::SerializeSeq>::end(seq_serializer)
    }
}

struct DummyLayer { input_dim: i64,  output_dim: i64 }

impl<'de> serde::Deserialize<'de> for DummyLayer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: serde::Deserializer<'de> {
        enum DummyLayerField { input_dim, output_dim }
        const DUMMY_LAYER_FIELDS: &[&str] = &["input_dim", "output_dim"];

        impl<'de> serde::Deserialize<'de> for DummyLayerField {
            fn deserialize<D>(deserializer: D) -> Result<DummyLayerField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = DummyLayerField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("expecting a field of `MilkshakeLayer`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<DummyLayerField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "input_dim" => Ok(DummyLayerField::input_dim),
                            "output_dim" => Ok(DummyLayerField::output_dim),
                            _ => Err(serde::de::Error::unknown_field(value, DUMMY_LAYER_FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)

            }
        }

        struct DummyLayerVisitor;

        impl<'de> serde::de::Visitor<'de> for DummyLayerVisitor {
            type Value = DummyLayer;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct MilkshakeLayer")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<DummyLayer, V::Error>
            where
                V: serde::de::SeqAccess<'de>,
            {
                let input_dim = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let output_dim = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;

                Ok(DummyLayer { input_dim, output_dim })
            }

            fn visit_map<V>(self, mut map: V) -> Result<DummyLayer, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut input_dim = None;
                let mut output_dim = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        DummyLayerField::input_dim => {
                            if input_dim.is_some() {
                                return Err(serde::de::Error::duplicate_field("input_dim"));
                            }

                            input_dim = Some(map.next_value()?);
                        }
                        DummyLayerField::output_dim => {
                            if output_dim.is_some() {
                                return Err(serde::de::Error::duplicate_field("output_dim"));
                            }
                            output_dim = Some(map.next_value()?);
                        }
                    }
                }
                let input_dim = input_dim.ok_or_else(|| serde::de::Error::missing_field("input_dim"))?;
                let output_dim = output_dim.ok_or_else(|| serde::de::Error::missing_field("output_dim"))?;

                Ok(DummyLayer { input_dim, output_dim })
            }
        }

        deserializer.deserialize_struct("MilkshakeLayer", DUMMY_LAYER_FIELDS, DummyLayerVisitor)
    }
}

pub struct Actor {
    pub vs: std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>,
    pub actor: MilkshakeNetwork,
    pub max_action: f64,
}

impl Actor {
    pub fn new(state_dim: i64, action_dim: i64, nn_shape: Vec<i64>, max_action: f64) -> Self {
        let vs = std::rc::Rc::new(std::cell::RefCell::new(tch::nn::VarStore::new(**device)));

        let mut shape = nn_shape.clone();
        shape.insert(0, state_dim);
        shape.insert(shape.len(), action_dim);

        let mut layers = Vec::new();

        for x in 1..shape.len() {
            layers.push(MilkshakeLayer {
                layer: tch::nn::linear(
                    vs.borrow().root(),
                    shape[x - 1],
                    shape[x],
                    Default::default(),
                ),

                input: shape[x - 1],
                output: shape[x],
            });
        }

        let actor = MilkshakeNetwork { layers };

        Actor {
            vs,
            actor,
            max_action,
        }
    }

    pub fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        <MilkshakeNetwork as tch::nn::Module>::forward(&self.actor, &xs)
    }
}

impl serde::Serialize for Actor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
        self.vs.borrow().save_to_stream(&mut cursor).expect("Failed to save actor varstore to byte buffer");

        let mut struct_serializer = serializer.serialize_struct("Actor", 3)?;

        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "actor_varstore", cursor.into_inner().as_slice())?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "actor_network", &self.actor)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "max_action", &self.max_action)?;

        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::end(struct_serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Actor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        enum ActorField { actor_varstore, actor_network, max_action }
        const ACTOR_FIELDS: &[&str] = &["actor_varstore", "actor_network", "max_action"];

        impl<'de> serde::Deserialize<'de> for ActorField {
            fn deserialize<D>(deserializer: D) -> Result<ActorField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = ActorField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("expecting a field of `actor`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<ActorField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "actor_varstore" => Ok(ActorField::actor_varstore),
                            "actor_network" => Ok(ActorField::actor_network),
                            "max_action" => Ok(ActorField::max_action),
                            _ => Err(serde::de::Error::unknown_field(value, ACTOR_FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)

            }
        }

        struct ActorVisitor;

        impl<'de> serde::de::Visitor<'de> for ActorVisitor {
            type Value = Actor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Actor")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Actor, V::Error>
            where
                V: serde::de::SeqAccess<'de>,
            {
                let actor_varstore: Vec<u8> = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let actor_network: Vec<DummyLayer> = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let max_action = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;

                let vs = std::rc::Rc::new(std::cell::RefCell::new(tch::nn::VarStore::new(**device)));

                let mut layers = Vec::new();
                for layer in actor_network {
                    let child = tch::nn::linear(vs.borrow().root(), layer.input_dim, layer.output_dim, Default::default());
                    layers.push(MilkshakeLayer {layer: child, input: layer.input_dim, output: layer.output_dim});
                }

                let cursor = std::io::Cursor::new(actor_varstore);
                vs.borrow_mut().load_from_stream(cursor).expect("Failed to load actor varstore from save file");

                let actor = MilkshakeNetwork { layers };

                Ok(Actor { vs, actor, max_action })
            }

            fn visit_map<V>(self, mut map: V) -> Result<Actor, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut max_action = None;
                let mut actor_network = None;
                let mut actor_varstore = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        ActorField::actor_varstore => {
                            if actor_varstore.is_some() {
                                return Err(serde::de::Error::duplicate_field("actor_varstore"));
                            }

                            actor_varstore = Some(map.next_value()?);
                        }

                        ActorField::actor_network => {
                            if actor_network.is_some() {
                                return Err(serde::de::Error::duplicate_field("actor_network"));
                            }

                            actor_network = Some(map.next_value()?);
                        }

                        ActorField::max_action => {
                            if max_action.is_some() {
                                return Err(serde::de::Error::duplicate_field("max_action"));
                            }

                            max_action = Some(map.next_value()?);
                        }
                    }
                }

                let actor_varstore: Vec<u8> = actor_varstore.ok_or_else(|| serde::de::Error::missing_field("actor_varstore"))?;
                let actor_network: Vec<DummyLayer> = actor_network.ok_or_else(|| serde::de::Error::missing_field("actor_network"))?;
                let max_action = max_action.ok_or_else(|| serde::de::Error::missing_field("max_action"))?;

                let vs = std::rc::Rc::new(std::cell::RefCell::new(tch::nn::VarStore::new(**device)));

                let mut layers = Vec::new();
                for layer in actor_network {
                    let child = tch::nn::linear(vs.borrow().root(), layer.input_dim, layer.output_dim, Default::default());
                    layers.push(MilkshakeLayer {layer: child, input: layer.input_dim, output: layer.output_dim});
                }

                let cursor = std::io::Cursor::new(actor_varstore);
                vs.borrow_mut().load_from_stream(cursor).expect("Failed to load actor varstore from save file");

                let actor = MilkshakeNetwork { layers };

                Ok(Actor { vs, actor, max_action })
            }
        }

        deserializer.deserialize_struct("Actor", ACTOR_FIELDS, ActorVisitor)
    }
}

pub struct Critic {
    pub vs: std::rc::Rc<std::cell::RefCell<tch::nn::VarStore>>,
    pub q1: MilkshakeNetwork,
    pub q2: MilkshakeNetwork,
}

impl Critic {
    pub fn new(state_dim: i64, action_dim: i64, q1_shape: Vec<i64>, q2_shape: Vec<i64>) -> Self {
        let vs = std::rc::Rc::new(std::cell::RefCell::new(tch::nn::VarStore::new(**device)));

        let mut q1_shape = q1_shape.clone();
        q1_shape.insert(0, state_dim + action_dim);
        q1_shape.insert(q1_shape.len(), 1);

        let mut q1_layers = Vec::new();

        for x in 1..q1_shape.len() {
            q1_layers.push(MilkshakeLayer {
                layer: tch::nn::linear(
                    vs.borrow().root(),
                    q1_shape[x - 1],
                    q1_shape[x],
                    Default::default(),
                ),

                input: q1_shape[x - 1],
                output: q1_shape[x],
            });
        }

        let mut q2_shape = q2_shape.clone();
        q2_shape.insert(0, state_dim + action_dim);
        q2_shape.insert(q2_shape.len(), 1);

        let mut q2_layers = Vec::new();

        for x in 1..q2_shape.len() {
            q2_layers.push(MilkshakeLayer {
                layer: tch::nn::linear(
                    vs.borrow().root(),
                    q2_shape[x - 1],
                    q2_shape[x],
                    Default::default(),
                ),

                input: q2_shape[x - 1],
                output: q2_shape[x],
            });
        }

        let q1 = MilkshakeNetwork { layers: q1_layers };
        let q2 = MilkshakeNetwork { layers: q2_layers };

        Critic { vs, q1, q2 }
    }

    pub fn forward(&self, state: &tch::Tensor, action: &tch::Tensor) -> (tch::Tensor, tch::Tensor) {
        let xs = tch::Tensor::cat(&[state, action], 1);

        let q1 = <MilkshakeNetwork as tch::nn::Module>::forward(&self.q1, &xs);
        let q2 = <MilkshakeNetwork as tch::nn::Module>::forward(&self.q2, &xs);

        (q1, q2)
    }

    pub fn Q1(&self, xs: &tch::Tensor) -> tch::Tensor {
        <MilkshakeNetwork as tch::nn::Module>::forward(&self.q1, &xs)
    }
}

impl serde::Serialize for Critic {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
        self.vs.borrow().save_to_stream(&mut cursor).expect("Failed to save critic varstore to byte buffer");

        let mut struct_serializer = serializer.serialize_struct("Critic", 3)?;

        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "critic_varstore", cursor.into_inner().as_slice())?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "q1_network", &self.q1)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "q2_network", &self.q2)?;

        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::end(struct_serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Critic {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        enum CriticField { critic_varstore, q1_network, q2_network }
        const CRITIC_FIELDS: &[&str] = &["critic_varstore", "q1_network", "q2_network"];

        impl<'de> serde::Deserialize<'de> for CriticField {
            fn deserialize<D>(deserializer: D) -> Result<CriticField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = CriticField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("expecting a field of `critic`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<CriticField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "critic_varstore" => Ok(CriticField::critic_varstore),
                            "q1_network" => Ok(CriticField::q1_network),
                            "q2_network" => Ok(CriticField::q2_network),
                            _ => Err(serde::de::Error::unknown_field(value, CRITIC_FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct CriticVisitor;

        impl<'de> serde::de::Visitor<'de> for CriticVisitor {
            type Value = Critic;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Critic")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Critic, V::Error>
            where
                V: serde::de::SeqAccess<'de>,
            {
                let critic_varstore: Vec<u8> = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let q1_network: Vec<DummyLayer> = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let q2_network: Vec<DummyLayer> = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;

                let vs = std::rc::Rc::new(std::cell::RefCell::new(tch::nn::VarStore::new(**device)));

                let mut q1_layers = Vec::new();
                for layer in q1_network {
                    let child = tch::nn::linear(vs.borrow().root(), layer.input_dim, layer.output_dim, Default::default());
                    q1_layers.push(MilkshakeLayer {layer: child, input: layer.input_dim, output: layer.output_dim});
                }

                let mut q2_layers = Vec::new();
                for layer in q2_network {
                    let child = tch::nn::linear(vs.borrow().root(), layer.input_dim, layer.output_dim, Default::default());
                    q2_layers.push(MilkshakeLayer {layer: child, input: layer.input_dim, output: layer.output_dim});
                }

                let cursor = std::io::Cursor::new(critic_varstore);
                vs.borrow_mut().load_from_stream(cursor).expect("Failed to load critic varstore from save file");

                let q1 = MilkshakeNetwork { layers: q1_layers };
                let q2 = MilkshakeNetwork { layers: q2_layers };

                Ok(Critic { vs, q1, q2 })
            }

            fn visit_map<V>(self, mut map: V) -> Result<Critic, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut critic_varstore = None;
                let mut q1_network = None;
                let mut q2_network = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        CriticField::critic_varstore => {
                            if critic_varstore.is_some() {
                                return Err(serde::de::Error::duplicate_field("critic_varstore"));
                            }

                            critic_varstore = Some(map.next_value()?);
                        }

                        CriticField::q1_network => {
                            if q1_network.is_some() {
                                return Err(serde::de::Error::duplicate_field("q1_network"));
                            }

                            q1_network = Some(map.next_value()?);
                        }

                        CriticField::q2_network => {
                            if q2_network.is_some() {
                                return Err(serde::de::Error::duplicate_field("q2_network"));
                            }

                            q2_network = Some(map.next_value()?);
                        }
                    }
                }

                let critic_varstore: Vec<u8> = critic_varstore.ok_or_else(|| serde::de::Error::missing_field("critic_varstore"))?;
                let q1_network: Vec<DummyLayer> = q1_network.ok_or_else(|| serde::de::Error::missing_field("q1_network"))?;
                let q2_network: Vec<DummyLayer> = q2_network.ok_or_else(|| serde::de::Error::missing_field("q2_network"))?;

                let vs = std::rc::Rc::new(std::cell::RefCell::new(tch::nn::VarStore::new(**device)));

                let mut q1_layers = Vec::new();
                for layer in q1_network {
                    let child = tch::nn::linear(vs.borrow().root(), layer.input_dim, layer.output_dim, Default::default());
                    q1_layers.push(MilkshakeLayer {layer: child, input: layer.input_dim, output: layer.output_dim});
                }

                let mut q2_layers = Vec::new();
                for layer in q2_network {
                    let child = tch::nn::linear(vs.borrow().root(), layer.input_dim, layer.output_dim, Default::default());
                    q2_layers.push(MilkshakeLayer {layer: child, input: layer.input_dim, output: layer.output_dim});
                }

                let cursor = std::io::Cursor::new(critic_varstore);
                vs.borrow_mut().load_from_stream(cursor).expect("Failed to load critic varstore from save file");

                let q1 = MilkshakeNetwork { layers: q1_layers };
                let q2 = MilkshakeNetwork { layers: q2_layers };

                Ok(Critic { vs, q1, q2 })
            }
        }

        deserializer.deserialize_struct("Critic", CRITIC_FIELDS, CriticVisitor)
    }
}

pub struct TD3 {
    actor: Actor,
    actor_target: Actor,
    critic: Critic,
    critic_target: Critic,

    actor_opt: Box<dyn MilkshakeOptimizer>,
    critic_opt: Box<dyn MilkshakeOptimizer>,

    pub action_dim: i64,
    pub state_dim: i64,
    pub max_action: f64,
    pub tau: f64,
    pub discount: f64,
    pub policy_noise: f64,
    pub noise_clip: f64,
    pub policy_freq: i64,
    pub total_it: i64,
}

impl TD3 {
    pub fn new(
        state_dim: i64,
        action_dim: i64,
        max_action: f64,
        actor_opt: &str,
        critic_opt: &str,
        actor_shape: Option<Vec<i64>>,
        q1_shape: Option<Vec<i64>>,
        q2_shape: Option<Vec<i64>>,
        tau: Option<f64>,
        discount: Option<f64>,
        policy_noise: Option<f64>,
        noise_clip: Option<f64>,
        policy_freq: Option<i64>,
    ) -> anyhow::Result<Self> {
        let actor_shape = actor_shape.unwrap_or(vec![64, 64]);
        let q1_shape = q1_shape.unwrap_or(vec![64, 64]);
        let q2_shape = q2_shape.unwrap_or(vec![64, 64]);

        let tau = tau.unwrap_or(0.005);
        let discount = discount.unwrap_or(0.99);
        let policy_noise = policy_noise.unwrap_or(0.2);
        let noise_clip = noise_clip.unwrap_or(0.5);
        let policy_freq = policy_freq.unwrap_or(2);

        let actor = Actor::new(state_dim, action_dim, actor_shape.clone(), max_action);
        let actor_target = Actor::new(state_dim, action_dim, actor_shape.clone(), max_action);

        let critic = Critic::new(state_dim, action_dim, q1_shape.clone(), q2_shape.clone());
        let critic_target = Critic::new(state_dim, action_dim, q1_shape.clone(), q2_shape.clone());

        let actor_opt: anyhow::Result<Box<dyn MilkshakeOptimizer>> = match actor_opt {
            "ADAM" => Ok(Box::new(ADAM::new(0.0003f64, actor.vs.clone()))),
            "CMAES" => Ok(Box::new(CMAES::new(actor.vs.clone(), None, None))),
            &_ => {
                anyhow::bail!("Invalid Actor Optimizer Chosen")
            }
        };

        let critic_opt: anyhow::Result<Box<dyn MilkshakeOptimizer>> = match critic_opt {
            "ADAM" => Ok(Box::new(ADAM::new(0.0003f64, critic.vs.clone()))),
            "CMAES" => Ok(Box::new(CMAES::new(critic.vs.clone(), None, None))),
            &_ => {
                anyhow::bail!("Invalid Critic Optimizer Chosen")
            }
        };

        let actor_opt = actor_opt?;
        let critic_opt = critic_opt?;

        Ok(TD3 {
            actor,
            actor_target,
            critic,
            critic_target,
            actor_opt,
            critic_opt,
            action_dim,
            state_dim,
            max_action,
            tau,
            discount,
            policy_noise,
            noise_clip,
            policy_freq,
            total_it: 0,
        })
    }

    pub fn select_action(&self, state: Vec<f64>) -> Vec<f64> {
        let state = tch::Tensor::from_slice(&state).to_device(**device);
        let tensor = self.actor.forward(&state).to_device(tch::Device::Cpu);
        let len = tensor.size().iter().fold(1, |sum, val| sum * *val as usize);

        let mut vec = vec![0f32; len];
        tensor.copy_data(vec.as_mut_slice(), len);

        vec.iter().map(|x| *x as f64).collect()
    }

    pub fn train(&mut self, replay_buffer: &ReplayBuffer, batch_size: Option<i64>) {
        let batch_size = batch_size.unwrap_or(256);
        let samples = replay_buffer.sample(batch_size);

        let state = &samples[0];
        let action = &samples[1];
        let next_state = &samples[2];
        let reward = &samples[3];
        let not_done = &samples[4];

        let target_q = tch::no_grad(|| {
            let noise =
                (action.rand_like() * self.policy_noise).clamp(-self.noise_clip, self.noise_clip);

            let next_action = (self.actor_target.forward(next_state) + noise)
                .clamp(-self.max_action, self.max_action);

            let q = self.critic_target.forward(next_state, &next_action);

            let target_q1 = &q.0;
            let target_q2 = &q.1;

            let min_q = target_q1.min_other(target_q2);

            reward.unsqueeze(1) + not_done.unsqueeze(1) * min_q * self.discount
        });

        let grads = self.critic_opt.grads();
        let mut critic_train_closure = || {
            let solutions = self.critic_opt.ask();
            let mut losses = vec![];

            for solution in &solutions {
                if !std::rc::Rc::ptr_eq(solution, &self.critic.vs) {
                    self.critic
                        .vs
                        .borrow_mut()
                        .copy(&solution.borrow())
                        .expect("Failed to copy test solution to critic");
                }

                let q = self.critic.forward(state, action);

                let current_q1 = &q.0;
                let current_q2 = &q.1;

                let q1_loss = current_q1.mse_loss(&target_q, tch::Reduction::Mean);
                let q2_loss = current_q2.mse_loss(&target_q, tch::Reduction::Mean);

                let critic_loss = q1_loss + q2_loss;
                losses.push(critic_loss);
            }

            self.critic_opt.tell(solutions, losses);

            let critic_result = self.critic_opt.result();
            if !std::rc::Rc::ptr_eq(&critic_result, &self.critic.vs) {
                self.critic
                    .vs
                    .borrow_mut()
                    .copy(&critic_result.borrow())
                    .expect("Failed to copy result to critic from optimizer");
            }
        };

        match grads {
            true => critic_train_closure(),
            false => tch::no_grad(critic_train_closure),
        }

        if self.total_it % self.policy_freq == 0 {
            let grads = self.actor_opt.grads();
            let mut actor_train_closure = || {
                let solutions = self.actor_opt.ask();
                let mut losses = vec![];

                for solution in &solutions {
                    if !std::rc::Rc::ptr_eq(solution, &self.actor.vs) {
                        self.actor
                            .vs
                            .borrow_mut()
                            .copy(&solution.borrow())
                            .expect("Failed to copy test solution to actor");
                    }

                    let loss = -1 * self
                        .critic
                        .Q1(&tch::Tensor::cat(&[state, &self.actor.forward(state)], 1))
                        .mean(tch::Kind::Float);

                    losses.push(loss);
                }

                self.actor_opt.tell(solutions, losses);

                let actor_result = self.actor_opt.result();
                if !std::rc::Rc::ptr_eq(&actor_result, &self.actor.vs) {
                    self.actor
                        .vs
                        .borrow_mut()
                        .copy(&actor_result.borrow())
                        .expect("Failed to copy result to actor from optimizer");
                }
            };

            match grads {
                true => actor_train_closure(),
                false => tch::no_grad(actor_train_closure),
            }

            tch::no_grad(|| {
                for (param, target_param) in self
                    .actor
                    .vs
                    .borrow_mut()
                    .trainable_variables()
                    .iter_mut()
                    .zip(
                        self.actor_target
                            .vs
                            .borrow_mut()
                            .trainable_variables()
                            .iter_mut(),
                    )
                {
                    target_param.copy_(
                        &(self.tau * param.copy() + (1f64 - self.tau) * target_param.copy()),
                    );
                }

                for (param, target_param) in self
                    .critic
                    .vs
                    .borrow_mut()
                    .trainable_variables()
                    .iter_mut()
                    .zip(
                        self.critic_target
                            .vs
                            .borrow_mut()
                            .trainable_variables()
                            .iter_mut(),
                    )
                {
                    target_param.copy_(
                        &(self.tau * param.copy() + (1f64 - self.tau) * target_param.copy()),
                    );
                }
            })
        }
    }
}

impl serde::Serialize for TD3 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut struct_serializer = serializer.serialize_struct("TD3", 13)?;

        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "actor", &self.actor)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "actor_target", &self.actor_target)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "critic", &self.critic)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "critic_target", &self.critic_target)?;

        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "action_dim", &self.action_dim)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "state_dim", &self.state_dim)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "max_action", &self.max_action)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "tau", &self.tau)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "discount", &self.discount)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "policy_noise", &self.policy_noise)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "noise_clip", &self.noise_clip)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "policy_freq", &self.policy_freq)?;
        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::serialize_field(&mut struct_serializer, "total_it", &self.total_it)?;

        <<S as serde::Serializer>::SerializeStruct as serde::ser::SerializeStruct>::end(struct_serializer)
    }
}

impl<'de> serde::Deserialize<'de> for TD3 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        enum TD3Field { actor, actor_target, critic, critic_target, action_dim, state_dim, max_action, tau, discount, policy_noise, noise_clip, policy_freq, total_it }
        const TD3_FIELDS: &[&str] = &["actor", "actor_target", "critic", "critic_target", "action_dim", "state_dim", "max_action", "tau", "discount", "policy_noise", "noise_clip", "policy_freq", "total_it"];

        impl<'de> serde::Deserialize<'de> for TD3Field {
            fn deserialize<D>(deserializer: D) -> Result<TD3Field, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = TD3Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("expecting a field of `TD3`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<TD3Field, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "actor" => Ok(TD3Field::actor),
                            "actor_target" => Ok(TD3Field::actor_target),
                            "critic" => Ok(TD3Field::critic),
                            "critic_target" => Ok(TD3Field::critic_target),

                            "action_dim" => Ok(TD3Field::action_dim),
                            "state_dim" => Ok(TD3Field::state_dim),
                            "max_action" => Ok(TD3Field::max_action),
                            "tau" => Ok(TD3Field::tau),
                            "discount" => Ok(TD3Field::discount),
                            "policy_noise" => Ok(TD3Field::policy_noise),
                            "noise_clip" => Ok(TD3Field::noise_clip),
                            "policy_freq" => Ok(TD3Field::policy_freq),
                            "total_it" => Ok(TD3Field::total_it),

                            _ => Err(serde::de::Error::unknown_field(value, TD3_FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct TD3Visitor;

        impl<'de> serde::de::Visitor<'de> for TD3Visitor {
            type Value = TD3;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct TD3")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<TD3, V::Error>
            where
                V: serde::de::SeqAccess<'de>,
            {
                let actor: Actor = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let actor_target: Actor = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let critic: Critic = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let critic_target: Critic = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;

                let action_dim = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(4, &self))?;
                let state_dim = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(5, &self))?;
                let max_action = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(6, &self))?;
                let tau = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(7, &self))?;
                let discount = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(8, &self))?;
                let policy_noise = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(9, &self))?;
                let noise_clip = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(10, &self))?;
                let policy_freq = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(11, &self))?;
                let total_it = seq.next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(12, &self))?;

                let actor_opt: Box<dyn MilkshakeOptimizer> = Box::new(ADAM::new(0.0003f64, actor.vs.clone()));
                let critic_opt: Box<dyn MilkshakeOptimizer> = Box::new(ADAM::new(0.0003f64, critic.vs.clone()));

                Ok(
                    TD3 {
                        actor,
                        actor_target,
                        critic,
                        critic_target,
                        actor_opt,
                        critic_opt,
                        action_dim,
                        state_dim,
                        max_action,
                        tau,
                        discount,
                        policy_noise,
                        noise_clip,
                        policy_freq,
                        total_it,
                    }
                )
            }

            fn visit_map<V>(self, mut map: V) -> Result<TD3, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut actor = None;
                let mut actor_target = None;
                let mut critic = None;
                let mut critic_target = None;

                let mut action_dim = None;
                let mut state_dim = None;
                let mut max_action = None;
                let mut tau = None;
                let mut discount = None;
                let mut policy_noise = None;
                let mut noise_clip = None;
                let mut policy_freq = None;
                let mut total_it = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        TD3Field::actor => {
                            if actor.is_some() {
                                return Err(serde::de::Error::duplicate_field("actor"));
                            }

                            actor = Some(map.next_value()?);
                        }

                        TD3Field::actor_target => {
                            if actor_target.is_some() {
                                return Err(serde::de::Error::duplicate_field("actor_target"));
                            }

                            actor_target = Some(map.next_value()?);
                        }

                        TD3Field::critic => {
                            if critic.is_some() {
                                return Err(serde::de::Error::duplicate_field("critic"));
                            }

                            critic = Some(map.next_value()?);
                        }

                        TD3Field::critic_target => {
                            if critic_target.is_some() {
                                return Err(serde::de::Error::duplicate_field("critic_target"));
                            }

                            critic_target = Some(map.next_value()?);
                        }

                        TD3Field::action_dim => {
                            if action_dim.is_some() {
                                return Err(serde::de::Error::duplicate_field("action_dim"));
                            }

                            action_dim = Some(map.next_value()?);
                        }

                        TD3Field::state_dim => {
                            if state_dim.is_some() {
                                return Err(serde::de::Error::duplicate_field("state_dim"));
                            }

                            state_dim = Some(map.next_value()?);
                        }

                        TD3Field::max_action => {
                            if max_action.is_some() {
                                return Err(serde::de::Error::duplicate_field("max_action"));
                            }

                            max_action = Some(map.next_value()?);
                        }

                        TD3Field::tau => {
                            if tau.is_some() {
                                return Err(serde::de::Error::duplicate_field("tau"));
                            }

                            tau = Some(map.next_value()?);
                        }

                        TD3Field::discount => {
                            if discount.is_some() {
                                return Err(serde::de::Error::duplicate_field("discount"));
                            }

                            discount = Some(map.next_value()?);
                        }

                        TD3Field::policy_noise => {
                            if policy_noise.is_some() {
                                return Err(serde::de::Error::duplicate_field("policy_noise"));
                            }

                            policy_noise = Some(map.next_value()?);
                        }

                        TD3Field::noise_clip => {
                            if noise_clip.is_some() {
                                return Err(serde::de::Error::duplicate_field("noise_clip"));
                            }

                            noise_clip = Some(map.next_value()?);
                        }

                        TD3Field::policy_freq => {
                            if policy_freq.is_some() {
                                return Err(serde::de::Error::duplicate_field("policy_freq"));
                            }

                            policy_freq = Some(map.next_value()?);
                        }

                        TD3Field::total_it => {
                            if total_it.is_some() {
                                return Err(serde::de::Error::duplicate_field("total_it"));
                            }

                            total_it = Some(map.next_value()?);
                        }
                    }
                }

                let actor: Actor = actor.ok_or_else(|| serde::de::Error::missing_field("actor"))?;
                let actor_target: Actor = actor_target.ok_or_else(|| serde::de::Error::missing_field("actor_target"))?;
                let critic: Critic = critic.ok_or_else(|| serde::de::Error::missing_field("critic"))?;
                let critic_target: Critic = critic_target.ok_or_else(|| serde::de::Error::missing_field("critic_target"))?;

                let actor_opt: Box<dyn MilkshakeOptimizer> = Box::new(ADAM::new(0.0003f64, actor.vs.clone()));
                let critic_opt: Box<dyn MilkshakeOptimizer> = Box::new(ADAM::new(0.0003f64, critic.vs.clone()));

                let action_dim = action_dim.ok_or_else(|| serde::de::Error::missing_field("action_dim"))?;
                let state_dim = state_dim.ok_or_else(|| serde::de::Error::missing_field("state_dim"))?;
                let max_action = max_action.ok_or_else(|| serde::de::Error::missing_field("max_action"))?;
                let tau = tau.ok_or_else(|| serde::de::Error::missing_field("tau"))?;
                let discount = discount.ok_or_else(|| serde::de::Error::missing_field("discount"))?;
                let policy_noise = policy_noise.ok_or_else(|| serde::de::Error::missing_field("policy_noise"))?;
                let noise_clip = noise_clip.ok_or_else(|| serde::de::Error::missing_field("noise_clip"))?;
                let policy_freq = policy_freq.ok_or_else(|| serde::de::Error::missing_field("policy_freq"))?;
                let total_it = total_it.ok_or_else(|| serde::de::Error::missing_field("total_it"))?;

                Ok(
                    TD3 {
                        actor,
                        actor_target,
                        critic,
                        critic_target,
                        actor_opt,
                        critic_opt,
                        action_dim,
                        state_dim,
                        max_action,
                        tau,
                        discount,
                        policy_noise,
                        noise_clip,
                        policy_freq,
                        total_it,
                    }
                )
            }
        }

        deserializer.deserialize_struct("TD3", TD3_FIELDS, TD3Visitor)
    }
}
