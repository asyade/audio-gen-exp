use std::sync::atomic::{AtomicBool, Ordering};
use crate::prelude::*;

use super::{RawAudioSample, SamplerNode};

/// TODO: rename to audio node host
#[derive(Clone)]
pub struct NodeHost {
    is_sync: Arc<AtomicBool>,
    payload: Arc<std::sync::RwLock<Option<RawAudioSample>>>,
    state: Option<HostState>,
}

#[derive(Clone)]
struct HostState {
    sampler: An<SamplerNode>,
}

impl NodeHost  {
    pub fn new() -> Self {
        Self {
            is_sync: Arc::new(AtomicBool::new(true)),
            payload: Arc::new(std::sync::RwLock::new(None)),
            state: None,
        }
    }

    pub fn load(&self, sample: RawAudioSample) -> CaResult<()> {
        let mut lock = self.payload.write().map_err(|_| CaError::Poisoned)?;
        lock.replace(sample);
        self.is_sync.store(false, std::sync::atomic::Ordering::SeqCst);
        drop(lock);
        Ok(())
    }

    pub fn fill(&mut self, buffer: &mut [&mut [f32]]) -> CaResult<()> {
        self.sync()?;
        if let Some(state) = self.state.as_mut() {
            state.fill(buffer);
        }
        Ok(())
    }

    fn sync(&mut self) -> CaResult<()> {
        if self.is_sync.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(())
        }
        let payload = self.payload.read().map_err(|_| CaError::Poisoned)?;
        match payload.as_ref() {
            Some(payload) => {
                self.state = Some(HostState::load(payload.clone())?);
                self.is_sync.store(true, Ordering::SeqCst);
                info!("sampler state loaded !");
            },
            None => {
                self.state = None;
                info!("sampler cleared !");
            }
        }
        Ok(())
    }
}

impl HostState {
    pub fn load(payload: RawAudioSample) -> CaResult<Self> {
        let sampler = An(SamplerNode::new(payload));
        Ok(Self {
            sampler,
        })
    }

    pub fn fill(&mut self, buffer: &mut [&mut [f32]]) {
        for x in 0..buffer[0].len() {
            let (l, r) = self.sampler.get_stereo();
            buffer[0][x] = l;
            buffer[1][x] = r;
        }
    }
}