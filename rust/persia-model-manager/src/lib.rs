#![allow(clippy::needless_return)]

use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use persia_libs::{
    anyhow::Error as AnyhowError,
    once_cell::sync::OnceCell,
    parking_lot::RwLock,
    rayon::{ThreadPool, ThreadPoolBuilder},
    serde::{self, Deserialize, Serialize},
    serde_yaml, thiserror, tracing,
};

use persia_embedding_config::{PersiaCommonConfig, PersiaGlobalConfigError, PersiaReplicaInfo};
use persia_embedding_holder::{
    array_linked_list::ArrayLinkedList, emb_entry::HashMapEmbeddingEntry, PersiaEmbeddingHolder,
    PersiaEmbeddingHolderError,
};
use persia_speedy::{Readable, Writable};
use persia_storage::{PersiaPath, PersiaPathImpl};

#[derive(Clone, Readable, Writable, thiserror::Error, Debug)]
pub enum EmbeddingModelManagerError {
    #[error("storage error {0}")]
    StorageError(String),
    #[error("embedding holder error: {0}")]
    PersiaEmbeddingHolderError(#[from] PersiaEmbeddingHolderError),
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("wait for other server time out when dump embedding")]
    WaitForOtherServerTimeOut,
    #[error("loading from an uncompelete embedding ckpt {0}")]
    LoadingFromUncompeleteCheckpoint(String),
    #[error("embedding file type worong")]
    WrongEmbeddingFileType,
    #[error("not ready error")]
    NotReadyError,
    #[error("failed to get embedding servers model manager status error")]
    FailedToGetStatus,
    #[error("failed to decode checkpoint info error {0}")]
    DecodeInfoError(String),
}

impl From<AnyhowError> for EmbeddingModelManagerError {
    fn from(e: AnyhowError) -> Self {
        let msg = format!("{:?}", e);
        EmbeddingModelManagerError::StorageError(msg)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(crate = "self::serde")]
pub struct EmbeddingModelInfo {
    pub num_shards: usize,
    pub num_internal_shards: usize,
    pub datetime: SystemTime,
}

#[derive(Clone, Readable, Writable, Debug)]
pub enum EmbeddingModelManagerStatus {
    Dumping(f32),
    Loading(f32),
    Idle,
    Failed(EmbeddingModelManagerError),
}

static EMBEDDING_MODEL_MANAGER: OnceCell<Arc<EmbeddingModelManager>> = OnceCell::new();

#[derive(Clone)]
pub struct EmbeddingModelManager {
    pub status: Arc<RwLock<EmbeddingModelManagerStatus>>,
    pub thread_pool: Arc<ThreadPool>,
    pub replica_index: usize,
    pub replica_size: usize,
}

impl EmbeddingModelManager {
    pub fn get() -> Result<Arc<Self>, EmbeddingModelManagerError> {
        let singleton = EMBEDDING_MODEL_MANAGER.get_or_try_init(|| {
            let common_config = PersiaCommonConfig::get()?;
            let replica_info = PersiaReplicaInfo::get()?;

            let singleton = Arc::new(Self::new(
                common_config.checkpointing_config.num_workers,
                replica_info.replica_index,
                replica_info.replica_size,
            ));
            Ok(singleton)
        });

        match singleton {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(e),
        }
    }

    fn new(concurrent_size: usize, replica_index: usize, replica_size: usize) -> Self {
        Self {
            status: Arc::new(RwLock::new(EmbeddingModelManagerStatus::Idle)),
            thread_pool: Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(concurrent_size)
                    .build()
                    .unwrap(),
            ),
            replica_index,
            replica_size,
        }
    }

    pub fn is_master_server(&self) -> bool {
        self.replica_index == 0
    }

    pub fn get_status(&self) -> EmbeddingModelManagerStatus {
        let status = self.status.read().clone();
        status
    }

    pub fn get_shard_dir(&self, root_dir: &PathBuf) -> PathBuf {
        let shard_dir_name = format!("s{}", self.replica_index);
        let shard_dir_name = PathBuf::from(shard_dir_name);
        let shard_dir = [root_dir, &shard_dir_name].iter().collect();
        shard_dir
    }

    pub fn get_other_shard_dir(&self, root_dir: &PathBuf, replica_index: usize) -> PathBuf {
        let shard_dir_name = format!("s{}", replica_index);
        let shard_dir_name = PathBuf::from(shard_dir_name);
        let shard_dir = [root_dir, &shard_dir_name].iter().collect();
        shard_dir
    }

    pub fn get_parent_dir(&self, root_dir: &PathBuf) -> PathBuf {
        let mut parent = root_dir.clone();
        parent.pop();
        parent
    }

    pub fn get_internam_shard_filename(&self, internal_shard_idx: usize) -> PathBuf {
        let file_name = format!(
            "replica_{}_shard_{}.emb",
            self.replica_index, internal_shard_idx
        );
        PathBuf::from(file_name)
    }

    pub fn get_emb_dump_done_file_name(&self) -> PathBuf {
        PathBuf::from("embedding_dump_done")
    }

    pub fn mark_embedding_dump_done(
        &self,
        emb_dir: PathBuf,
        num_internal_shards: usize,
    ) -> Result<(), EmbeddingModelManagerError> {
        tracing::info!("mark_embedding_dump_done {:?}", emb_dir);
        let emb_dump_done_file = self.get_emb_dump_done_file_name();
        let emb_dump_done_path = PersiaPath::from_vec(vec![&emb_dir, &emb_dump_done_file]);
        emb_dump_done_path.create(false)?;

        let model_info = EmbeddingModelInfo {
            num_shards: self.replica_size,
            num_internal_shards,
            datetime: SystemTime::now(),
        };

        let s = serde_yaml::to_string(&model_info).expect("failed to serialize model info to yaml");
        emb_dump_done_path.append(s)?;

        Ok(())
    }

    pub fn check_embedding_dump_done(
        &self,
        emb_dir: &PathBuf,
    ) -> Result<bool, EmbeddingModelManagerError> {
        let emb_dump_done_file = self.get_emb_dump_done_file_name();
        let emb_dump_done_path = PersiaPath::from_vec(vec![emb_dir, &emb_dump_done_file]);
        Ok(emb_dump_done_path.is_file()?)
    }

    pub fn load_embedding_checkpoint_info(
        &self,
        emb_dir: &PathBuf,
    ) -> Result<EmbeddingModelInfo, EmbeddingModelManagerError> {
        let emb_dump_done_file = self.get_emb_dump_done_file_name();
        let emb_dump_done_path = PersiaPath::from_vec(vec![emb_dir, &emb_dump_done_file]);
        let s: String = emb_dump_done_path.read_to_string()?;
        tracing::debug!("load_embedding_checkpoint_info {}", s);

        serde_yaml::from_str(&s)
            .map_err(|e| EmbeddingModelManagerError::DecodeInfoError(format!("{:?}", e)))
    }

    pub fn waiting_for_all_embedding_server_dump(
        &self,
        timeout_sec: usize,
        dst_dir: PathBuf,
    ) -> Result<(), EmbeddingModelManagerError> {
        let replica_size = self.replica_size;
        if replica_size < 2 {
            tracing::info!("replica_size < 2, will not wait for other embedding servers");
            return Ok(());
        }
        tracing::info!("start to wait for all embedding server dump compelete");
        let start_time = std::time::Instant::now();
        let mut compeleted = std::collections::HashSet::with_capacity(replica_size);
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            for replica_index in 0..replica_size {
                if compeleted.contains(&replica_index) {
                    continue;
                }
                let shard_dir = self.get_other_shard_dir(&dst_dir, replica_index);
                let done = self.check_embedding_dump_done(&shard_dir)?;
                if done {
                    tracing::info!("dump complete for index {}", replica_index);
                    compeleted.insert(replica_index);
                } else {
                    tracing::info!("waiting dump emb for index {}...", replica_index);
                }
            }
            if compeleted.len() == replica_size {
                tracing::info!("all embedding server compelte to dump embedding");
                break;
            }

            if start_time.elapsed().as_secs() as usize > timeout_sec {
                tracing::error!("waiting for other embedding server to dump embedding TIMEOUT");
                return Err(EmbeddingModelManagerError::WaitForOtherServerTimeOut);
            }
        }

        Ok(())
    }

    pub fn dump_internal_shard_embeddings(
        &self,
        internal_shard_idx: usize,
        dst_dir: PathBuf,
        embedding_holder: PersiaEmbeddingHolder,
    ) -> Result<(), EmbeddingModelManagerError> {
        let shard = embedding_holder
            .get_shard_by_index(internal_shard_idx)
            .read();

        let file_name = self.get_internam_shard_filename(internal_shard_idx);
        let emb_path = PersiaPath::from_vec(vec![&dst_dir, &file_name]);

        emb_path.write_all_speedy(&shard.linkedlist)?;

        Ok(())
    }

    pub fn load_internal_shard_embeddings(
        &self,
        file_path: PathBuf,
        embedding_holder: PersiaEmbeddingHolder,
    ) -> Result<(), EmbeddingModelManagerError> {
        let decoded = self.load_array_linked_list(file_path)?;

        decoded.into_iter().for_each(|entry| {
            let sign = entry.sign();
            let mut shard = embedding_holder.shard(&sign).write();
            shard.insert(sign, entry);
        });

        Ok(())
    }

    pub fn load_array_linked_list(
        &self,
        file_path: PathBuf,
    ) -> Result<ArrayLinkedList<HashMapEmbeddingEntry>, EmbeddingModelManagerError> {
        tracing::debug!("loading embedding linked list from {:?}", file_path);
        let emb_path = PersiaPath::from_pathbuf(file_path);
        let decoded: ArrayLinkedList<HashMapEmbeddingEntry> = emb_path.read_to_end_speedy()?;
        Ok(decoded)
    }

    pub fn dump_embedding(
        &self,
        dst_dir: PathBuf,
        embedding_holder: PersiaEmbeddingHolder,
    ) -> Result<(), EmbeddingModelManagerError> {
        *self.status.write() = EmbeddingModelManagerStatus::Dumping(0.0);
        tracing::info!("start to dump embedding to {:?}", dst_dir);

        let shard_dir = self.get_shard_dir(&dst_dir);
        let num_internal_shards = embedding_holder.num_internal_shards();
        let num_dumped_shards = Arc::new(AtomicUsize::new(0));
        let manager = Self::get()?;

        (0..num_internal_shards).for_each(|internal_shard_idx| {
            let dst_dir = shard_dir.clone();
            let num_dumped_shards = num_dumped_shards.clone();
            let manager = manager.clone();
            let embedding_holder = embedding_holder.clone();

            self.thread_pool.spawn(move || {
                let closure = || -> Result<(), EmbeddingModelManagerError> {
                    manager.dump_internal_shard_embeddings(
                        internal_shard_idx,
                        dst_dir.clone(),
                        embedding_holder,
                    )?;

                    let dumped = num_dumped_shards.fetch_add(1, Ordering::AcqRel) + 1;
                    let dumping_progress = (dumped as f32) / (num_internal_shards as f32);

                    *manager.status.write() =
                        EmbeddingModelManagerStatus::Dumping(dumping_progress);
                    tracing::debug!("dumping progress is {}", dumping_progress);

                    if dumped >= num_internal_shards {
                        manager.mark_embedding_dump_done(dst_dir.clone(), num_internal_shards)?;
                        if manager.is_master_server() {
                            let parent_dir = manager.get_parent_dir(&dst_dir);
                            manager
                                .waiting_for_all_embedding_server_dump(600, parent_dir.clone())?;
                            manager.mark_embedding_dump_done(parent_dir, num_internal_shards)?;
                        }

                        tracing::info!("dump embedding to {:?} compelete", dst_dir);
                        *manager.status.write() = EmbeddingModelManagerStatus::Idle;
                    }

                    Ok(())
                };

                if let Err(e) = closure() {
                    *manager.status.write() = EmbeddingModelManagerStatus::Failed(e);
                }
            })
        });

        Ok(())
    }

    pub fn get_emb_file_list_in_dir(
        &self,
        dir: PathBuf,
    ) -> Result<Vec<PathBuf>, EmbeddingModelManagerError> {
        let done = self.check_embedding_dump_done(&dir)?;
        if !done {
            return Err(
                EmbeddingModelManagerError::LoadingFromUncompeleteCheckpoint(format!("{:?}", &dir)),
            );
        }
        let persia_dir = PersiaPath::from_pathbuf(dir.clone());
        let file_list = persia_dir.list()?;
        tracing::debug!("file_list is {:?}", file_list);

        let file_list: Vec<PathBuf> = file_list
            .into_iter()
            .filter(|x| x.extension() == Some(OsStr::new("emb")))
            .collect();
        tracing::debug!("file_list end with emb is {:?}", file_list);

        if file_list.len() == 0 {
            tracing::error!("trying to load embedding from an empty dir");
            return Err(
                EmbeddingModelManagerError::LoadingFromUncompeleteCheckpoint(format!("{:?}", &dir)),
            );
        }

        Ok(file_list)
    }

    pub fn load_embedding_from_dir(
        &self,
        dir: PathBuf,
        embedding_holder: PersiaEmbeddingHolder,
    ) -> Result<(), EmbeddingModelManagerError> {
        tracing::info!("start to load embedding from dir {:?}", dir);

        let file_list = self.get_emb_file_list_in_dir(dir)?;

        let num_total_files = file_list.len();
        let num_loaded_files = Arc::new(AtomicUsize::new(0));

        for file_path in file_list.into_iter() {
            tracing::debug!("spawn to load embedding from {:?}", file_path);
            let manager = Self::get()?;

            self.thread_pool.spawn({
                let manager = manager.clone();
                let num_loaded_files = num_loaded_files.clone();
                let embedding_holder = embedding_holder.clone();
                move || {
                    let closure = || -> Result<(), EmbeddingModelManagerError> {
                        tracing::debug!("start to execute load embedding from {:?}", file_path);
                        manager
                            .load_internal_shard_embeddings(file_path.clone(), embedding_holder)?;

                        let loaded = num_loaded_files.fetch_add(1, Ordering::AcqRel) + 1;
                        let loading_progress = (loaded as f32) / (num_total_files as f32);
                        *manager.status.write() =
                            EmbeddingModelManagerStatus::Loading(loading_progress);
                        tracing::debug!("load embedding progress is {}", loading_progress);

                        if num_total_files == loaded {
                            *manager.status.write() = EmbeddingModelManagerStatus::Idle;
                            let parent = manager.get_parent_dir(&file_path);
                            tracing::info!("load checkpoint from {:?} compelete", parent);
                        }

                        Ok(())
                    };

                    if let Err(e) = closure() {
                        *manager.status.write() = EmbeddingModelManagerStatus::Failed(e);
                    }
                }
            });
        }
        *self.status.write() = EmbeddingModelManagerStatus::Loading(0.0);

        Ok(())
    }
}
