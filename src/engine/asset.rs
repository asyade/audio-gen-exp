
use crate::prelude::*;

#[derive(Debug, Clone)]
pub enum Asset {
    Tmp {
        path: PathBuf,
        directory: Arc<RwLock<TempDir>>,
    },
    Stored {
        path: PathBuf,
    }
}

impl PartialEq for Asset {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Asset::Stored { path: left }, Asset::Stored { path: right }) if right == left => {
                true
            },
            (Asset::Tmp { path: left, .. }, Asset::Tmp { path: right, .. }) if right == left => {
                true
            },
            _ => {
                false
            }
        }
    }
}

impl Asset {

    pub fn load_naive(path: PathBuf) -> Asset {
        Self::Stored { path: path.to_owned() }
    }

    pub fn path(&self) -> &Path {
        match self {
            Asset::Stored { path } => &path,
            Asset::Tmp { path, .. } => &path,
        }
    }

    pub async fn create_tmp(extension: &str, directory: Arc<RwLock<TempDir>>) -> Self {
        let fname = Local::now().format("%Y-%m-%d_%H-%M-%S");
        let path = directory.read().await.path().join(format!("{}.{}", fname, extension));
        Self::Tmp {
            path,
            directory,
        }
    }

    pub async fn copy_to(&self, path: &Path) -> CaResult<Asset> {
        tokio::fs::copy(self.path(), path).await?;
        Ok(Asset::Stored { path: path.to_owned() })
    }

    /* IMPORTANT: this is not using async I/O and will cause issues */
    pub async fn get_samples(&self) -> CaResult<Vec<f32>> {
        let mut reader = hound::WavReader::open(self.path())?;
        let samples: Vec<i16> = reader.samples::<i16>().try_collect()?;
        Ok(samples.into_iter().map(|s| s as f32 / i16::MAX as f32).collect())
    }
}