use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::{
    collections::HashMap, ffi::OsStr, path::PathBuf, process::Stdio,
    sync::atomic::AtomicUsize,
};

use crate::prelude::*;
use serde::de::DeserializeOwned;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::select;
use tokio::{
    net::TcpListener,
    process::Command,
    sync::{
        mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender},
        OwnedRwLockWriteGuard,
    },
};
use tokio_stream::StreamExt;
use tokio_util::codec::{BytesCodec, FramedRead};
use tracing::instrument;

use super::diffusion::{DiffusionModelOpt, StringOpt};
use super::{
    asset::Asset,
    diffusion::DiffusionModelTemplate,
};

//@TODO
const HF_API_KEY: &str = "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";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "id")]
pub enum FromProcessRequest {
    Ack {
        request: serde_json::Value,
    },
    Log {
        message: String,
        level: String,
    },
    CallBack {
        call_id: usize,
        payload: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "id")]
pub enum ToProcessRequest {
    Call {
        procedure_id: &'static str,
        /// Id of the specific call
        call_id: usize,
        /// Payload passed to python procedure
        payload: serde_json::Value,
    },
}

#[derive(Clone)]
pub struct CondaExecutor {
    entrypoint: PathBuf,
    hf_api_key: String,

    remote_process: Arc<RwLock<RemoteProcess>>,

    ipc_proxy: UnboundedSender<FromProcessRequest>,
    last_call: Arc<AtomicUsize>,
}

#[derive(Debug, Clone)]
struct RemoteProcess {
    alive: Arc<AtomicBool>,
    marshalled: Arc<RwLock<HashMap<usize, MarshalledCall>>>,
    to_socket: UnboundedSender<ToProcessRequest>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct MarshalledCall {
    /// Maintain the caller idle until the call is resolved
    lock: OwnedRwLockWriteGuard<serde_json::Value>,
    call_id: usize,
}

pub trait CondaExecutorTask: Serialize {
    type Result: DeserializeOwned;
    const ID: &'static str;
}

#[derive(Deserialize, Serialize)]
pub struct GetDiffusionModelTemplateTask {
    pub model: String,
}
impl CondaExecutorTask for GetDiffusionModelTemplateTask {
    type Result = DiffusionModelTemplate;
    const ID: &'static str = "GetDiffusionModelTemplateTask";
}

#[derive(Deserialize, Serialize)]
pub struct RunDiffusionModelTemplateTask {
    pub template: DiffusionModelTemplate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunDiffusionModelTemplateTaskResult {
    error: Option<String>,
    assets: Option<Vec<String>>,
}

impl CondaExecutorTask for RunDiffusionModelTemplateTask {
    type Result = RunDiffusionModelTemplateTaskResult;
    const ID: &'static str = "RunDiffusionModelTemplateTask";
}


#[derive(Deserialize, Serialize)]
pub struct GetAvailableDiffusionModel {
}

impl CondaExecutorTask for GetAvailableDiffusionModel {
    type Result = Vec<String>;
    const ID: &'static str = "GetAvailableDiffusionModel";
}

impl CondaExecutor {
    pub async fn init(cache_directory: PathBuf) -> CaResult<Self> {
        let entrypoint = cache_directory.join("conda_executor.py");
        let _ = tokio::fs::remove_file(&entrypoint);
        tokio::fs::write(&entrypoint, include_str!("./conda_executor.py")).await?;
        let (ipc_proxy, recv) = unbounded_channel();
        let instance = Self {
            remote_process: Arc::new(RwLock::new(
                RemoteProcess::spawn(
                    HF_API_KEY.to_string(),
                    entrypoint.to_str().unwrap().to_owned(),
                )
                .await?,
            )),
            ipc_proxy,
            hf_api_key: HF_API_KEY.to_string(),
            entrypoint,
            last_call: Arc::new(AtomicUsize::new(0)),
        };
        Ok(instance)
    }

    /// TODO: resurect not alive background processs
    /// TODO: avoid useless value cast here and there, we can use appropriate deserializer directly (value suck for large payload)
    pub async fn call<T: CondaExecutorTask>(&self, task: T) -> CaResult<T::Result> {
        let mut remote_process_lock = self.remote_process.write().await;
        let serialized = serde_json::to_value(task)?;
        match remote_process_lock
            .call_wait(
                self.last_call
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                T::ID,
                serialized.clone(),
            )
            .await
        {
            Ok(value) => Ok(serde_json::from_value(value)?),
            Err(e) => {
                warn!("resurecting conda process ...");
                *remote_process_lock = RemoteProcess::spawn(
                    HF_API_KEY.to_string(),
                    self.entrypoint.to_str().unwrap().to_owned(),
                )
                .await?;
                let value = remote_process_lock
                    .call_wait(
                        self.last_call
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                        T::ID,
                        serialized,
                    )
                    .await?;
                Ok(serde_json::from_value(value)?)
            }
        }
    }

    pub async fn get_available_diffusion_model(&self) -> CaResult<Vec<String>> {
        self.call(GetAvailableDiffusionModel{}).await
    }

    pub async fn process_diffusion_model(
        &self,
        mut template: DiffusionModelTemplate,
        output_directory: PathBuf,
    ) -> CaResult<Vec<Asset>> {
        template.options.insert(
            "HF_API_KEY".to_string(),
            DiffusionModelOpt::String(StringOpt {
                value: Some(HF_API_KEY.to_string()),
                ..Default::default()
            }),
        );
        template.options.insert(
            "OUTPUT_DIRECTORY".to_string(),
            DiffusionModelOpt::String(StringOpt {
                value: Some(output_directory.to_str().unwrap().to_string()),
                ..Default::default()
            }),
        );
        let output = self.call(RunDiffusionModelTemplateTask { template })
            .await?;
        match output {
            RunDiffusionModelTemplateTaskResult { assets: Some(assets), error: None } => {
                let assets = assets.into_iter().map(|path| {
                    Asset::Stored { path: path.into() }
                }).collect();
                Ok(assets)
            },
            RunDiffusionModelTemplateTaskResult { error, ..} => Err(CaError::ExternalError { message: error }),
        }
    }
}

impl RemoteProcess {
    pub async fn spawn(api_key: String, entry_point: String) -> CaResult<Self> {
        let (to_socket, proxy) = unbounded_channel();
        let instance = RemoteProcess {
            to_socket,
            alive: Arc::new(AtomicBool::new(true)),
            marshalled: Arc::new(RwLock::new(HashMap::new())),
        };

        let mut nretry = 0;
        let mut start_port = 4240;
        loop {
            match TcpListener::bind(SocketAddr::new("127.0.0.1".parse().unwrap(), start_port)).await
            {
                Ok(listener) => {
                    tokio::spawn(
                        instance
                            .clone()
                            .handle_process(api_key, entry_point, start_port),
                    );
                    tokio::spawn(instance.clone().handle_server(listener, proxy));
                    return Ok(instance);
                }
                Err(e) => {
                    nretry += 1;
                    start_port += 1;
                    if nretry > 256 {
                        error!("Failed to find free port !");
                        return Err(e.into());
                    } else {
                        warn!(port = start_port, "port unavailable: {:?}", e);
                    }
                }
            }
        }
    }

    async fn handle_process(self, api_key: String, entry_point: String, port: u16) {
        let result = run_python_script_with_conda_env(
            &entry_point,
            "audiocraft_env",
            vec![("HF_API_KEY", api_key.clone()), ("PORT", port.to_string())],
        )
        .await;
        warn!("Python process exited: {:?}", result);
        self.down().await;
    }

    async fn handle_server(
        self,
        listener: TcpListener,
        proxy: UnboundedReceiver<ToProcessRequest>,
    ) {
        if let Ok((connection, _)) = listener.accept().await {
            info!("Process connected !");
            if let Err(e) = self.clone().handle_stream(connection, proxy).await {
                error!("connection down: {}", e);
            }
        } else {
            error!("Process never connect !");
        }
        self.down().await;
    }

    async fn handle_stream(
        self,
        mut connection: TcpStream,
        mut proxy: UnboundedReceiver<ToProcessRequest>,
    ) -> CaResult<()> {
        let (read, mut write) = connection.split();
        let mut framed = FramedRead::new(read, BytesCodec::new());
        let mut buffer: Vec<u8> = Vec::with_capacity(1024 * 1024);
        loop {
            select! {
                from_socket = framed.next() => {
                    match from_socket {
                        Some(Ok(bytes)) => {
                            buffer.extend(bytes);

                            while buffer.len() > 4 {
                                let packet_len = i32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);

                                if buffer.len() - 4 < packet_len as usize {
                                    continue;
                                }

                                let range = 4 .. (packet_len as usize + 4);

                                match serde_json::from_slice::<FromProcessRequest>(&buffer[range.clone()]) {
                                    Ok(packet) => {
                                        if let Err(e) = self.handle_packet(packet).await {
                                            warn!("failed to handle packet: {:?}", e);
                                        }
                                    },
                                    Err(e) => {
                                        let raw = String::from_utf8_lossy(&buffer[range]).to_string();
                                        error!(packet=raw, packet_len=packet_len, "invalide packet: {:?}", e)
                                    }
                                }
                                buffer = Vec::from(&buffer[packet_len as usize + 4..]);
                            }
                        }
                        Some(Err(err)) => println!("Socket closed with error: {:?}", err),
                        _ => {}
                    }
                }
                from_proxy = proxy.recv() => {
                    match from_proxy {
                        Some(request) => {
                            let packet = serde_json::to_vec(&request).unwrap();
                            let packet: Vec<u8> = [&(packet.len() as i32).to_be_bytes(), &packet[..]].concat();
                            write.write_all(&packet[..]).await?;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    async fn handle_packet(&self, packet: FromProcessRequest) -> CaResult<()> {
        match packet {
            FromProcessRequest::Ack { .. } => Ok(()),
            FromProcessRequest::Log { message, level } => {
                if level.to_lowercase() == "error" {
                    error!(from = "conda executor", level = level, "{}", message);
                } else {
                    info!(from = "conda executor", level = level, "{}", message);
                }
                Ok(())
            }
            FromProcessRequest::CallBack { call_id, payload } => {
                info!(call_id = call_id, "got callback");
                match self.marshalled.write().await.remove(&call_id) {
                    Some(mut marshall) => {
                        info!(call_id = call_id, payload = ?payload, "marshalled callback resolved");
                        *marshall.lock = payload;
                    }
                    None => {
                        error!("retrive callback value but the marshalled call does not exists !");
                    }
                }

                Ok(())
            }
        }
    }

    #[instrument(skip(self))]
    async fn call_wait(
        &self,
        call_id: usize,
        procedure_id: &'static str,
        payload: serde_json::Value,
    ) -> CaResult<serde_json::Value> {
        let mut global_lock = self.marshalled.write().await;

        if !self.alive.load(std::sync::atomic::Ordering::SeqCst) {
            error!("cant marshall request, the RemoteProcess is not alive");
            return Err(CaError::NotAlive);
        }

        let locked = Arc::new(RwLock::new(serde_json::Value::Null));
        let lock = locked.clone().write_owned().await;
        let call = MarshalledCall {
            call_id: call_id,
            lock,
        };

        global_lock.insert(call.call_id, call);

        let _ = self.to_socket.send(ToProcessRequest::Call {
            procedure_id,
            call_id,
            payload: payload,
        });

        drop(global_lock);

        let mut lock = locked.write().await;
        Ok(lock.take())
    }

    async fn down(&self) {
        let mut tasks = self.marshalled.write().await;

        for (_task, marshall) in tasks.drain() {
            error!(call_id = marshall.call_id, "marshalled call canceled");
        }

        if !self
            .alive
            .fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |_x| Some(false),
            )
            .unwrap()
        {
            return;
        }
        info!("cleaning pending tasks ...");
    }
}

pub async fn run_python_script_with_conda_env<I, K, V>(
    script_path: &str,
    conda_env: &str,
    env: I,
) -> CaResult<String>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<OsStr>,
    V: AsRef<OsStr>,
{
    let conda_activate_script = "C:\\Users\\corbe\\anaconda3\\Scripts\\activate.bat";

    // Command to activate the Conda environment and run the Python script
    let cmd = format!(
        "call {} {} && python {}",
        conda_activate_script, conda_env, script_path
    );

    tracing::info!(cmd = cmd, "spawning conda executor");
    // Run the command in a new Command process
    let output = Command::new("cmd")
        .args(&["/C", &cmd])
        .envs(env)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await?;

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let stderr_str = String::from_utf8_lossy(&output.stdout);

    if output.status.success() {
        Ok(stdout_str.to_string())
    } else {
        let stdout_str: &str = &stdout_str;
        let stderr_str: &str = &stderr_str;

        error!(
            stdout = stdout_str,
            stderr = stderr_str,
            "conda runtime failed with status {}",
            output.status
        );
        Err(CaError::UnexpectedExit(output.status))
    }
}
