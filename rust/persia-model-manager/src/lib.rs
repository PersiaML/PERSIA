#![allow(clippy::needless_return)]

use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use persia_libs::{
    anyhow::Error as AnyhowError,
    bincode,
    once_cell::sync::OnceCell,
    parking_lot::RwLock,
    rayon::{ThreadPool, ThreadPoolBuilder},
    thiserror, tracing,
};

use persia_embedding_config::{
    PerisaJobType, PersiaCommonConfig, PersiaEmbeddingServerConfig, PersiaGlobalConfigError,
    PersiaReplicaInfo,
};
use persia_embedding_holder::{
    array_linked_list::ArrayLinkedList, emb_entry::HashMapEmbeddingEntry, PersiaEmbeddingHolder,
    PersiaEmbeddingHolderError,
};
use persia_speedy::{Readable, Writable};
use persia_storage::{PersiaPath, PersiaPathImpl};

#[derive(Clone, Readable, Writable, thiserror::Error, Debug)]
pub enum PersistenceManagerError {
    #[error("storage error")]
    StorageError(String),
    #[error("embedding holder error: {0}")]
    PersiaEmbeddingHolderError(#[from] PersiaEmbeddingHolderError),
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("wait for other server time out when dump embedding")]
    WaitForOtherServerTimeOut,
    #[error("loading from an uncompelete embedding ckpt")]
    LoadingFromUncompeleteCheckpoint,
    #[error("embedding file type worong")]
    WrongEmbeddingFileType,
    #[error("not ready error")]
    NotReadyError,
    #[error("failed to get status error")]
    FailedToGetStatus,
}

impl From<AnyhowError> for PersistenceManagerError {
    fn from(e: AnyhowError) -> Self {
        let msg = format!("{:?}", e);
        PersistenceManagerError::StorageError(msg)
    }
}

#[derive(Clone, Readable, Writable, Debug)]
pub enum PersiaPersistenceStatus {
    Dumping(f32),
    Loading(f32),
    Idle,
    Failed(PersistenceManagerError),
}

static MODEL_PERSISTENCE_MANAGER: OnceCell<Arc<PersiaPersistenceManager>> = OnceCell::new();

#[derive(Clone)]
pub struct PersiaPersistenceManager {
    embedding_holder: PersiaEmbeddingHolder,
    status: Arc<RwLock<PersiaPersistenceStatus>>,
    thread_pool: Arc<ThreadPool>,
    sign_per_file: usize,
    replica_index: usize,
    replica_size: usize,
    cur_task: PerisaJobType,
}

impl PersiaPersistenceManager {
    pub fn get() -> Result<Arc<Self>, PersistenceManagerError> {
        let singleton = MODEL_PERSISTENCE_MANAGER.get_or_try_init(|| {
            let server_config = PersiaEmbeddingServerConfig::get()?;
            let common_config = PersiaCommonConfig::get()?;
            let embedding_holder = PersiaEmbeddingHolder::get()?;
            let replica_info = PersiaReplicaInfo::get()?;

            let singleton = Arc::new(Self::new(
                embedding_holder,
                server_config.num_persistence_workers,
                server_config.num_signs_per_file,
                replica_info.replica_index,
                replica_info.replica_size,
                common_config.job_type.clone(),
            ));
            Ok(singleton)
        });

        match singleton {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(e),
        }
    }

    fn new(
        embedding_holder: PersiaEmbeddingHolder,
        concurrent_size: usize,
        sign_per_file: usize,
        replica_index: usize,
        replica_size: usize,
        cur_task: PerisaJobType,
    ) -> Self {
        Self {
            embedding_holder,
            status: Arc::new(RwLock::new(PersiaPersistenceStatus::Idle)),
            thread_pool: Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(concurrent_size)
                    .build()
                    .unwrap(),
            ),
            sign_per_file,
            replica_index,
            replica_size,
            cur_task,
        }
    }

    pub fn is_master_server(&self) -> bool {
        self.replica_index == 0
    }

    pub fn get_status(&self) -> PersiaPersistenceStatus {
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

    pub fn get_upper_dir(&self, root_dir: &PathBuf) -> PathBuf {
        let mut upper = root_dir.clone();
        upper.pop();
        upper
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
    ) -> Result<(), PersistenceManagerError> {
        let emb_dump_done_file = self.get_emb_dump_done_file_name();
        let emb_dump_done_path = PersiaPath::from_vec(vec![&emb_dir, &emb_dump_done_file]);
        emb_dump_done_path.create(false)?;
        Ok(())
    }

    pub fn check_embedding_dump_done(
        &self,
        emb_dir: &PathBuf,
    ) -> Result<bool, PersistenceManagerError> {
        let emb_dump_done_file = self.get_emb_dump_done_file_name();
        let emb_dump_done_path = PersiaPath::from_vec(vec![emb_dir, &emb_dump_done_file]);
        let res = emb_dump_done_path.is_file()?;
        Ok(res)
    }

    pub fn waiting_for_all_embedding_server_dump(
        &self,
        timeout_sec: usize,
        dst_dir: PathBuf,
    ) -> Result<(), PersistenceManagerError> {
        let replica_size = self.replica_size;
        if replica_size < 2 {
            tracing::info!("replica_size < 2, will not wait for other embedding servers");
            return Ok(());
        }
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
                return Err(PersistenceManagerError::WaitForOtherServerTimeOut);
            }
        }

        Ok(())
    }

    pub fn dump_internal_shard_embeddings(
        &self,
        internal_shard_idx: usize,
        dst_dir: PathBuf,
    ) -> Result<(), PersistenceManagerError> {
        let encoded = {
            let shard = self
                .embedding_holder
                .get_shard_by_index(internal_shard_idx)
                .read();
            let array_linked_list = &shard.linkedlist;
            let encoded: Vec<u8> = bincode::serialize(&array_linked_list).unwrap();
            encoded
        };

        let file_name = self.get_internam_shard_filename(internal_shard_idx);
        let emb_path = PersiaPath::from_vec(vec![&dst_dir, &file_name]);

        emb_path.write_all(encoded)?;

        Ok(())
    }

    pub fn load_internal_shard_embeddings(
        &self,
        file_path: PathBuf,
    ) -> Result<(), PersistenceManagerError> {
        let emb_path = PersiaPath::from_pathbuf(file_path);
        let bytes: Vec<u8> = emb_path.read_to_end()?;
        let decoded: ArrayLinkedList<HashMapEmbeddingEntry> =
            bincode::deserialize(&bytes[..]).unwrap();

        decoded.into_iter().for_each(|entry| {
            let sign = entry.sign();
            let mut shard = self.embedding_holder.shard(&sign).write();
            shard.insert(sign, entry);
        });

        Ok(())
    }

    pub fn dump_embedding(&self, dst_dir: PathBuf) -> Result<(), PersistenceManagerError> {
        *self.status.write() = PersiaPersistenceStatus::Dumping(0.0);
        tracing::info!("start to dump embedding to {:?}", dst_dir);

        let shard_dir = self.get_shard_dir(&dst_dir);
        let num_internal_shards = self.embedding_holder.num_internal_shards();
        let num_dumped_shards = Arc::new(AtomicUsize::new(0));
        let manager = Self::get()?;

        (0..num_internal_shards).for_each(|internal_shard_idx| {
            let dst_dir = shard_dir.clone();
            let num_dumped_shards = num_dumped_shards.clone();
            let manager = manager.clone();

            self.thread_pool.spawn(move || {
                let closure = || -> Result<(), PersistenceManagerError> {
                    manager.dump_internal_shard_embeddings(internal_shard_idx, dst_dir.clone())?;

                    let dumped = num_dumped_shards.fetch_add(1, Ordering::AcqRel) + 1;
                    let dumping_progress = (dumped as f32) / (num_internal_shards as f32);

                    *manager.status.write() = PersiaPersistenceStatus::Dumping(dumping_progress);
                    tracing::debug!("dumping progress is {}", dumping_progress);

                    if dumped >= num_internal_shards {
                        manager.mark_embedding_dump_done(dst_dir.clone())?;
                        if manager.is_master_server() {
                            let upper_dir = manager.get_upper_dir(&dst_dir);
                            manager
                                .waiting_for_all_embedding_server_dump(600, upper_dir.clone())?;
                            manager.mark_embedding_dump_done(upper_dir)?;

                            *manager.status.write() = PersiaPersistenceStatus::Idle;
                        }
                    }

                    Ok(())
                };

                if let Err(e) = closure() {
                    *manager.status.write() = PersiaPersistenceStatus::Failed(e);
                }
            })
        });

        Ok(())
    }

    pub fn load_embedding_from_dir(&self, dst_dir: PathBuf) -> Result<(), PersistenceManagerError> {
        tracing::info!("start to load embedding from dir {:?}", dst_dir);
        let done = self.check_embedding_dump_done(&dst_dir)?;
        if !done {
            tracing::error!("trying to load embedding from uncompelete checkpoint");
            return Err(PersistenceManagerError::LoadingFromUncompeleteCheckpoint);
        }
        let dst_dir = self.get_shard_dir(&dst_dir);
        let done = self.check_embedding_dump_done(&dst_dir)?;
        if !done {
            return Err(PersistenceManagerError::LoadingFromUncompeleteCheckpoint);
        }
        let dst_dir = PersiaPath::from_pathbuf(dst_dir);
        let file_list = dst_dir.list()?;
        tracing::debug!("file_list is {:?}", file_list);

        let file_list: Vec<PathBuf> = file_list
            .into_iter()
            .filter(|x| x.extension() == Some(OsStr::new("emb")))
            .collect();
        tracing::debug!("file_list end with emb is {:?}", file_list);

        if file_list.len() == 0 {
            tracing::error!("trying to load embedding from an empty dir");
            return Err(PersistenceManagerError::LoadingFromUncompeleteCheckpoint);
        }

        let num_total_files = file_list.len();
        let num_loaded_files = Arc::new(AtomicUsize::new(0));

        for file in file_list.into_iter() {
            self.load_embedding_from_file(file, num_loaded_files.clone(), num_total_files)?;
        }

        *self.status.write() = PersiaPersistenceStatus::Loading(0.0);

        Ok(())
    }

    pub fn load_embedding_from_file(
        &self,
        file_path: PathBuf,
        num_loaded_files: Arc<AtomicUsize>,
        num_total_files: usize,
    ) -> Result<(), PersistenceManagerError> {
        tracing::debug!("spawn to load embedding from {:?}", file_path);
        let manager = Self::get()?;

        self.thread_pool.spawn({
            let manager = manager.clone();
            move || {
                let closure = || -> Result<(), PersistenceManagerError> {
                    tracing::debug!("start to execute load embedding from {:?}", file_path);
                    manager.load_internal_shard_embeddings(file_path)?;

                    let loaded = num_loaded_files.fetch_add(1, Ordering::AcqRel) + 1;
                    let loading_progress = (loaded as f32) / (num_total_files as f32);
                    *manager.status.write() = PersiaPersistenceStatus::Loading(loading_progress);
                    tracing::debug!("load embedding progress is {}", loading_progress);

                    if num_total_files == loaded {
                        *manager.status.write() = PersiaPersistenceStatus::Idle;
                    }

                    Ok(())
                };

                if let Err(e) = closure() {
                    *manager.status.write() = PersiaPersistenceStatus::Failed(e);
                }
            }
        });

        Ok(())
    }
}
