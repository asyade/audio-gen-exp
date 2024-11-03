use std::process::ExitStatus;

use crate::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CaError {
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error(transparent)]
    Serde(#[from]serde_json::Error),
    #[error("unexpected exit code: {0}")]
    UnexpectedExit(ExitStatus),
    #[error("not alive")]
    NotAlive,
    #[error("poisoned mutex")]
    Poisoned,
    #[error("asset not found: {0}")]
    AssetNotFound(usize),
    #[error("hound: {0}")]
    HoundError(#[from] hound::Error),
    #[error("expected non empty value")]
    Empty,
    #[error("incorect buffer configuration")]
    CoruptedBuffer,
    #[error("externaly occured error: {message:?}")]
    ExternalError {
        message: Option<String>,
    },
}

pub type CaResult<T> = Result<T, CaError>;