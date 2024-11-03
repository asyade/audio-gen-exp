use crate::prelude::*;

macro_rules! impl_as_variant {
    ($variant:pat, $enum:path, $value_type:path) => {
        paste::paste!{
            impl $enum {
                pub fn [<as_ $variant:lower _ref>](&self) -> Option< &[<$variant Opt>] >{
                    match self {
                        Self::$variant(opt) => Some(opt),
                        _ => None,
                    }
                }
                pub fn [<as_ $variant:lower _mut>](&mut self) -> Option< &mut [<$variant Opt>] >{
                    match self {
                        Self::$variant(opt) => Some(opt),
                        _ => None,
                    }
                }

                pub fn [<unwrap_ $variant:lower _ref>](&self) -> Option< &[<$variant Opt>] >{
                    match self {
                        Self::$variant(opt) => Some(opt),
                        _ => panic!("option not expected"),
                    }
                }
                pub fn [<unwrap_ $variant:lower _mut>](&mut self) -> Option< &mut [<$variant Opt>] >{
                    match self {
                        Self::$variant(opt) => Some(opt),
                        _ => panic!("option not expected"),
                    }
                }

                pub fn [<as_ $variant:lower _value_ref>](&self) -> Option< & $value_type >{
                    match self {
                        Self::$variant( [<$variant Opt>] { value, .. }) => value.as_ref(),
                        _ => None,
                    }
                }

                pub fn [<as_ $variant:lower _value_mut>](&mut self) -> Option< &mut  $value_type >{
                    match self {
                        Self::$variant( [<$variant Opt>] { value, .. }) => value.as_mut(),
                        _ => None,
                    }
                }


                pub fn [<as_ $variant:lower _value>](&self) -> Option<  $value_type >{
                    match self {
                        Self::$variant( [<$variant Opt>] { value, .. }) => value.clone(),
                        _ => None,
                    }
                }

            }
        }
    };
}

#[derive(Debug, Clone, Serialize, Deserialize, derive_more::Unwrap, PartialEq)]
#[serde(tag = "kind")]
pub enum DiffusionModelOpt {
    Int(IntOpt),
    Float(FloatOpt),
    String(StringOpt)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntOpt {
    pub range: Option<Range<i64>>,
    pub value: Option<i64>,
    #[serde(default)]
    pub hidden: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FloatOpt {
    pub range: Option<Range<f32>>,
    pub value: Option<f32>,
    #[serde(default)]
    pub hidden: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct StringOpt {
    pub possible_values: Option<Vec<String>>,
    pub max_length: Option<String>,
    pub value: Option<String>,
    #[serde(default)]
    pub hidden: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiffusionModelTemplate {
    pub options: HashMap<String, DiffusionModelOpt>,
}

impl_as_variant!(Int, DiffusionModelOpt, i64);
impl_as_variant!(Float, DiffusionModelOpt, f32);
impl_as_variant!(String, DiffusionModelOpt, String);

impl DiffusionModelOpt {
    pub fn into_raw_value(self) -> Option<String> {
        match self {
            DiffusionModelOpt::Int(IntOpt { value: Some(value) , .. }) => Some(format!("{}", value)),
            DiffusionModelOpt::Float(FloatOpt { value: Some(value) , .. }) => Some(format!("{}", value)),
            DiffusionModelOpt::String(StringOpt { value: Some(value) , .. }) => Some(format!("{}", value)),
            _ => None,
        }
    }

    pub fn hidden(&self) -> bool {
        match self {
            DiffusionModelOpt::Int(IntOpt { hidden, .. }) => *hidden,
            DiffusionModelOpt::Float(FloatOpt { hidden, .. }) => *hidden,
            DiffusionModelOpt::String(StringOpt { hidden, .. }) => *hidden,
        }
    }

}