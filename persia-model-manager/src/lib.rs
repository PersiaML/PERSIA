#![allow(clippy::needless_return)]
use anyhow::Error as AnyhowError;
use once_cell::sync::OnceCell;

use parking_lot::RwLock;
use persia_embedding_config::{
    PerisaIntent, PersiaCommonConfig, PersiaGlobalConfigError, PersiaPersistenceStorage,
    PersiaReplicaInfo, PersiaShardedServerConfig,
};
use persia_embedding_datatypes::HashMapEmbeddingEntry;
use persia_embedding_holder::{PersiaEmbeddingHolder, PersiaEmbeddingHolderError};
use persia_full_amount_manager::{FullAmountManager, PersiaFullAmountManagerError};
use persia_speedy::{Readable, Writable};
use persia_storage_visitor::{
    PersiaCephVisitor, PersiaHdfsVisitor, PersiaStorageVisitor, SpeedyObj,
};
use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Readable, Writable, Error, Debug)]
pub enum PersistenceManagerError {
    #[error("storage error")]
    StorageError(String),
    #[error("full amount manager error: {0}")]
    PersiaFullAmountManagerError(#[from] PersiaFullAmountManagerError),
    #[error("embedding holder error: {0}")]
    PersiaEmbeddingHolderError(#[from] PersiaEmbeddingHolderError),
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("wait for other shard time out when dump embedding")]
    WaitForOtherShardTimeOut,
    #[error("loading from an uncompelete embedding ckpt")]
    LoadingFromUncompeleteCheckpoint,
    #[error("embedding file type worong")]
    WrongEmbeddingFileType,
    #[error("not ready error")]
    NotReadyError,
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
    Failed(String),
}

static MODEL_PERSISTENCE_MANAGER: OnceCell<Arc<PersiaPersistenceManager>> = OnceCell::new();

#[derive(Clone)]
pub struct PersiaPersistenceManager {
    storage_visitor: Arc<dyn PersiaStorageVisitor>,
    embedding_holder: PersiaEmbeddingHolder,
    full_amount_manager: Arc<FullAmountManager>,
    status: Arc<parking_lot::RwLock<PersiaPersistenceStatus>>,
    thread_pool: Arc<rayon::ThreadPool>,
    sign_per_file: usize,
    shard_idx: usize,
    shard_num: usize,
    cur_task: PerisaIntent,
}

impl PersiaPersistenceManager {
    pub fn get() -> Result<Arc<Self>, PersistenceManagerError> {
        let singleton = MODEL_PERSISTENCE_MANAGER.get_or_try_init(|| {
            let server_config = PersiaShardedServerConfig::get()?;
            let common_config = PersiaCommonConfig::get()?;
            let embedding_holder = PersiaEmbeddingHolder::get()?;
            let full_amount_manager = FullAmountManager::get()?;
            let replica_info = PersiaReplicaInfo::get()?;

            let storage_visitor: Arc<dyn PersiaStorageVisitor> = match server_config.storage {
                PersiaPersistenceStorage::Ceph => Arc::new(PersiaCephVisitor {}),
                PersiaPersistenceStorage::Hdfs => Arc::new(PersiaHdfsVisitor::new()),
            };

            let singleton = Arc::new(Self::new(
                storage_visitor,
                embedding_holder,
                full_amount_manager,
                server_config.num_persistence_workers,
                server_config.num_signs_per_file,
                replica_info.replica_index,
                replica_info.replica_size,
                common_config.intent.clone(),
            ));
            Ok(singleton)
        });

        match singleton {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(e),
        }
    }

    fn new(
        storage_visitor: Arc<dyn PersiaStorageVisitor>,
        embedding_holder: PersiaEmbeddingHolder,
        full_amount_manager: Arc<FullAmountManager>,
        concurrent_size: usize,
        sign_per_file: usize,
        shard_idx: usize,
        shard_num: usize,
        cur_task: PerisaIntent,
    ) -> Self {
        Self {
            storage_visitor,
            embedding_holder,
            full_amount_manager,
            status: Arc::new(parking_lot::RwLock::new(PersiaPersistenceStatus::Idle)),
            thread_pool: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(concurrent_size)
                    .build()
                    .unwrap(),
            ),
            sign_per_file,
            shard_idx,
            shard_num,
            cur_task,
        }
    }

    pub fn get_status(&self) -> PersiaPersistenceStatus {
        let status = self.status.read().clone();
        status
    }

    pub fn get_shard_dir(&self, root_dir: &PathBuf) -> PathBuf {
        let shard_dir_name = format!("s{}", self.shard_idx);
        let shard_dir_name = PathBuf::from(shard_dir_name);
        let shard_dir = [root_dir, &shard_dir_name].iter().collect();
        shard_dir
    }

    pub fn get_other_shard_dir(&self, root_dir: &PathBuf, shard_idx: usize) -> PathBuf {
        let shard_dir_name = format!("s{}", shard_idx);
        let shard_dir_name = PathBuf::from(shard_dir_name);
        let shard_dir = [root_dir, &shard_dir_name].iter().collect();
        shard_dir
    }

    pub fn get_upper_dir(&self, root_dir: &PathBuf) -> PathBuf {
        let mut upper = root_dir.clone();
        upper.pop();
        upper
    }

    pub fn get_full_amount_embedding(
        &self,
    ) -> Result<Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)>, PersistenceManagerError> {
        let res = self.full_amount_manager.keys_values();
        Ok(res)
    }

    pub fn get_done_file_name(&self) -> PathBuf {
        PathBuf::from("embedding_dump_done")
    }

    pub fn mark_embedding_dump_done(
        &self,
        emb_dir: PathBuf,
    ) -> Result<(), PersistenceManagerError> {
        let done_file = self.get_done_file_name();
        self.storage_visitor.create_file(emb_dir, done_file)?;
        Ok(())
    }

    pub fn check_embedding_dump_done(
        &self,
        emb_dir: &PathBuf,
    ) -> Result<bool, PersistenceManagerError> {
        let done_file = self.get_done_file_name();
        let done_path: PathBuf = [emb_dir, &done_file].iter().collect();
        let res = self.storage_visitor.is_file(done_path)?;
        Ok(res)
    }

    pub fn waiting_for_all_sharded_server_dump(
        &self,
        timeout_sec: usize,
        dst_dir: PathBuf,
    ) -> Result<(), PersistenceManagerError> {
        let num_total_shard = self.shard_num;
        if num_total_shard < 2 {
            tracing::info!("num_total_shard < 2, will not wait for other sharded servers");
            return Ok(());
        }
        let start_time = std::time::Instant::now();
        let mut compeleted = std::collections::HashSet::with_capacity(num_total_shard);
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            for shard_idx in 0..num_total_shard {
                if compeleted.contains(&shard_idx) {
                    continue;
                }
                let shard_dir = self.get_other_shard_dir(&dst_dir, shard_idx);
                let done = self.check_embedding_dump_done(&shard_dir)?;
                if done {
                    tracing::info!("dump complete for shard {}", shard_idx);
                    compeleted.insert(shard_idx);
                } else {
                    tracing::info!("waiting dump emb for shard {}...", shard_idx);
                }
            }
            if compeleted.len() == num_total_shard {
                tracing::info!("all sharded server compelte to dump embedding");
                break;
            }

            if start_time.elapsed().as_secs() as usize > timeout_sec {
                tracing::error!("waiting for other sharded server to dump embedding TIMEOUT");
                return Err(PersistenceManagerError::WaitForOtherShardTimeOut);
            }
        }

        Ok(())
    }

    pub fn dump_embedding_segment(
        &self,
        dst_dir: PathBuf,
        segment: Vec<(u64, Arc<parking_lot::RwLock<HashMapEmbeddingEntry>>)>,
        file_index: usize,
        num_dumped_signs: Arc<AtomicUsize>,
        num_total_signs: usize,
    ) -> Result<(), PersistenceManagerError> {
        tracing::debug!("spawn embedding segment for file_index {}", file_index);
        let manager = Self::get();
        if manager.is_err() {
            tracing::error!("failed to get persistence manager");
            return Err(PersistenceManagerError::NotReadyError);
        }
        let manager = manager.unwrap();
        self.thread_pool.spawn({
            let manager = manager.clone();
            move || {
                tracing::debug!("start to execute dump embedding segment for file_index {}", file_index);
                let segment_size = segment.len();
                let segment_content: Vec<(u64, HashMapEmbeddingEntry)> = segment
                    .into_iter()
                    .map(|(id, emb)| {
                        (id.clone(), emb.read().clone())
                    })
                    .collect();

                let speedy_content = SpeedyObj::EmbeddingVec(segment_content);
                let date = chrono::Local::now()
                    .format("%Y-%m-%d-%H-%M-%S")
                    .to_string();

                let file_name = format!("{}_{}_{}.emb", date, manager.shard_idx, file_index);
                let file_name = PathBuf::from(file_name);
                if let Err(e) = manager.storage_visitor.dump_to_file_speedy(speedy_content, dst_dir.clone(), file_name) {
                    let msg = format!("{:?}", e);
                    tracing::error!("dump embedding error: {}", msg);
                    *manager.status.write() = PersiaPersistenceStatus::Failed(msg);
                }
                else {
                    let dumped = num_dumped_signs.fetch_add(segment_size, Ordering::AcqRel);
                    let cur_dumped = dumped + segment_size;

                    let dumping_progress = (cur_dumped as f32) / (num_total_signs as f32);
                    *manager.status.write() = PersiaPersistenceStatus::Dumping(dumping_progress);
                    tracing::debug!("dumping progress is {}", dumping_progress);

                    if cur_dumped >= num_total_signs {
                        tracing::debug!("cur_dumped >= num_total_signs, cur_dumped is {}, num_total_signs is {}", cur_dumped, num_total_signs);
                        if let Err(e) = manager.mark_embedding_dump_done(dst_dir.clone()) {
                            let msg = format!("{:?}", e);
                            tracing::error!("dump embedding error: {}", msg);
                            *manager.status.write() = PersiaPersistenceStatus::Failed(msg);
                        }
                        else {
                            if manager.shard_idx == 0 {
                                let upper_dir = manager.get_upper_dir(&dst_dir);
                                if let Err(e) = manager.waiting_for_all_sharded_server_dump(600, upper_dir.clone()) {
                                    let msg = format!("{:?}", e);
                                    tracing::error!("dump embedding error: {}", msg);
                                    *manager.status.write() = PersiaPersistenceStatus::Failed(msg);
                                }
                                else {
                                    if let Err(e) = manager.mark_embedding_dump_done(upper_dir) {
                                        let msg = format!("{:?}", e);
                                        tracing::error!("failed to mark embedding done file: {}", msg);
                                        *manager.status.write() = PersiaPersistenceStatus::Failed(msg);
                                    }
                                    else {
                                        *manager.status.write() = PersiaPersistenceStatus::Idle;
                                    }
                                }
                            }
                        }
                    }
                    else {
                        tracing::debug!("cur_dumped < num_total_signs, cur_dumped is {}, num_total_signs is {}", cur_dumped, num_total_signs);
                    }
                }
            }
        });

        Ok(())
    }

    pub fn dump_full_amount_embedding(
        &self,
        dst_dir: PathBuf,
    ) -> Result<(), PersistenceManagerError> {
        let mut embeddings = self.get_full_amount_embedding()?;
        let num_total_signs = embeddings.len();
        if num_total_signs > 0 {
            *self.status.write() = PersiaPersistenceStatus::Dumping(0.0);
            tracing::info!("start to dump embedding to {:?}", dst_dir);

            let shard_dir = self.get_shard_dir(&dst_dir);
            let mut file_index: usize = 0;
            let num_dumped_signs = Arc::new(AtomicUsize::new(0));

            while self.sign_per_file < embeddings.len() {
                let mut segment = embeddings.split_off(self.sign_per_file);
                std::mem::swap(&mut embeddings, &mut segment);
                self.dump_embedding_segment(
                    shard_dir.clone(),
                    segment,
                    file_index,
                    num_dumped_signs.clone(),
                    num_total_signs,
                )?;
                file_index += 1;
            }

            self.dump_embedding_segment(
                shard_dir.clone(),
                embeddings,
                file_index,
                num_dumped_signs.clone(),
                num_total_signs,
            )?;
        }

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
        let file_list = self.storage_visitor.list_dir(dst_dir.clone())?;
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

        Ok(())
    }

    pub fn load_embedding_from_file(
        &self,
        file_path: PathBuf,
        num_loaded_files: Arc<AtomicUsize>,
        num_total_files: usize,
    ) -> Result<(), PersistenceManagerError> {
        tracing::debug!("spawn to load embedding from {:?}", file_path);
        let manager = Self::get();
        if manager.is_err() {
            tracing::error!("failed to get persistence manager");
            return Err(PersistenceManagerError::NotReadyError);
        }
        let manager = manager.unwrap();
        self.thread_pool.spawn({
            let manager = manager.clone();
            move || {
                tracing::debug!("start to execute load embedding from {:?}", file_path);
                let speedy_content = manager.storage_visitor.read_from_file_speedy(file_path);
                if speedy_content.is_err() {
                    let msg = String::from("failed to read from file speedy");
                    tracing::error!("{}", msg);
                    *manager.status.write() = PersiaPersistenceStatus::Failed(msg);
                } else {
                    let speedy_content = speedy_content.unwrap();
                    match speedy_content {
                        SpeedyObj::EmbeddingVec(content) => {
                            let load_opt = match manager.cur_task {
                                PerisaIntent::Train | PerisaIntent::Eval => true,
                                PerisaIntent::Infer(_) => false,
                            };
                            let embeddings = manager.wrap_embeddings(content, load_opt);

                            let weak_ptrs = embeddings
                                .iter()
                                .map(|(k, v)| (k.clone(), Arc::downgrade(v)))
                                .collect();
                            if let Ok(_) = manager.full_amount_manager.commit_weak_ptrs(weak_ptrs) {
                                for (id, entry) in embeddings.into_iter() {
                                    manager.embedding_holder.inner.insert(id, entry);
                                }

                                let cur_loaded_files =
                                    num_loaded_files.fetch_add(1, Ordering::AcqRel);
                                let cur_loaded_files = cur_loaded_files + 1;

                                let loading_progress =
                                    (cur_loaded_files as f32) / (num_total_files as f32);
                                *manager.status.write() =
                                    PersiaPersistenceStatus::Loading(loading_progress);
                                tracing::debug!("load embedding progress is {}", loading_progress);

                                if num_total_files == cur_loaded_files {
                                    *manager.status.write() = PersiaPersistenceStatus::Idle;
                                }
                            } else {
                                let msg = String::from(
                                    "failed to commit embedding to full amount manager",
                                );
                                tracing::error!("{}", msg);
                                *manager.status.write() = PersiaPersistenceStatus::Failed(msg);
                            }
                        }
                        _ => {
                            let msg = String::from("wrong embedding file type");
                            tracing::error!("{}", msg);
                            *manager.status.write() = PersiaPersistenceStatus::Failed(msg);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    pub fn wrap_embeddings(
        &self,
        content: Vec<(u64, HashMapEmbeddingEntry)>,
        load_opt: bool,
    ) -> Vec<(u64, Arc<parking_lot::RwLock<HashMapEmbeddingEntry>>)> {
        let embeddings: Vec<(u64, Arc<parking_lot::RwLock<HashMapEmbeddingEntry>>)> = if load_opt {
            content
                .into_iter()
                .map(|(id, entry)| (id, Arc::new(parking_lot::RwLock::new(entry))))
                .collect()
        } else {
            let emb: Vec<(u64, HashMapEmbeddingEntry)> = content
                .into_iter()
                .map(|(id, entry)| {
                    let emb_entry = HashMapEmbeddingEntry::from_emb_infer(entry.emb().to_vec());
                    (id, emb_entry)
                })
                .collect();

            emb.into_iter()
                .map(|(id, entry)| (id, Arc::new(parking_lot::RwLock::new(entry))))
                .collect()
        };
        embeddings
    }
}
