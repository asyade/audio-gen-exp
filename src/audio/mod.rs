use typenum::Same;

use crate::prelude::*;

pub mod node_host;

#[derive(Clone)]
pub struct SamplerNode {
    repeat: bool,
    index: usize,
    buffer: Vec<(f32, f32)>
}

#[derive(Clone)]
pub enum RawAudioSample {
    Mono(Vec<f32>),
    Stereo(Vec<f32>, Vec<f32>),
}

impl RawAudioSample {
    pub fn stereoify(self) -> Vec<(f32, f32)> {
        match self {
            RawAudioSample::Mono(signal) => signal.into_iter().map(|sig| (sig, sig)).collect(),
            RawAudioSample::Stereo(left, right) => left.into_iter().zip(right.into_iter()).collect(),
        }
    }
}

impl SamplerNode {
    pub fn new(sample: RawAudioSample) -> Self {
        let mut buffer = sample.stereoify();
        if buffer.len() == 0 {
            buffer = vec![(0.0, 0.0)];
        }
        SamplerNode {
            repeat: false,
            index: 0,
            buffer,
        }
    }
}

impl AudioNode for SamplerNode {
    const ID: u64 = 4242;
    type Inputs = U0;
    type Outputs = U2;

    #[inline]
    fn tick(&mut self, input: &Frame<f32, Self::Inputs>) -> Frame<f32, Self::Outputs> {
        let (left, right) = self.buffer[self.index];
        self.index += 1;
        if self.index == self.buffer.len() {
            self.index = 0;
        }
        [left, right].into()
    }
}
