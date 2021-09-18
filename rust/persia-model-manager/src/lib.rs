#![allow(clippy::needless_return)]

use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use persia_libs::{
    anyhow::Error as AnyhowError,
    chrono,
    once_cell::sync::OnceCell,
    parking_lot::RwLock,
    rayon::{ThreadPool, ThreadPoolBuilder},
    thiserror, tracing,
};

use persia_common::HashMapEmbeddingEntry;
use persia_embedding_config::{
    PerisaJobType, PersiaCommonConfig, PersiaEmbeddingServerConfig, PersiaGlobalConfigError,
    PersiaReplicaInfo,
};
use persia_embedding_holder::{PersiaEmbeddingHolder, PersiaEmbeddingHolderError};
use persia_full_amount_manager::{FullAmountManager, PersiaFullAmountManagerError};
use persia_speedy::{Readable, Writable};
use persia_storage::{PersiaPath, PersiaPathImpl};

#[derive(Readable, Writable, thiserror::Error, Debug)]
pub enum PersistenceManagerError {
    #[error("storage error")]
    StorageError(String),
    #[error("full amount manager error: {0}")]
    PersiaFullAmountManagerError(#[from] PersiaFullAmountManagerError),
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
    embedding_holder: PersiaEmbeddingHolder,
    full_amount_manager: Arc<FullAmountManager>,
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
            let full_amount_manager = FullAmountManager::get()?;
            let replica_info = PersiaReplicaInfo::get()?;

            let singleton = Arc::new(Self::new(
                embedding_holder,
                full_amount_manager,
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
        full_amount_manager: Arc<FullAmountManager>,
        concurrent_size: usize,
        sign_per_file: usize,
        replica_index: usize,
        replica_size: usize,
        cur_task: PerisaJobType,
    ) -> Self {
        Self {
            embedding_holder,
            full_amount_manager,
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

    pub fn get_full_amount_embedding(
        &self,
    ) -> Result<Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)>, PersistenceManagerError> {
        let res = self.full_amount_manager.keys_values();
        Ok(res)
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

    pub fn dump_embedding_segment(
        &self,
        dst_dir: PathBuf,
        segment: Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)>,
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

                let date = chrono::Local::now()
                    .format("%Y-%m-%d-%H-%M-%S")
                    .to_string();

                let file_name = format!("{}_{}_{}.emb", date, manager.replica_index, file_index);
                let file_name = PathBuf::from(file_name);
                let emb_path = PersiaPath::from_vec(vec![&dst_dir, &file_name]);
                if let Err(e) = emb_path.write_all_speedy(segment_content) {
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
                            if manager.replica_index == 0 {
                                let upper_dir = manager.get_upper_dir(&dst_dir);
                                if let Err(e) = manager.waiting_for_all_embedding_server_dump(600, upper_dir.clone()) {
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
                let file_path = PersiaPath::from_pathbuf(file_path);
                let content: Result<Vec<(u64, HashMapEmbeddingEntry)>, _> =
                    file_path.read_to_end_speedy();
                if content.is_err() {
                    let msg = String::from("failed to read from file speedy");
                    tracing::error!("{}", msg);
                    *manager.status.write() = PersiaPersistenceStatus::Failed(msg);
                } else {
                    let content = content.unwrap();
                    let load_opt = match manager.cur_task {
                        PerisaJobType::Train | PerisaJobType::Eval => true,
                        PerisaJobType::Infer => false,
                    };
                    let embeddings = manager.wrap_embeddings(content, load_opt);

                    let weak_ptrs = embeddings
                        .iter()
                        .map(|(k, v)| (k.clone(), Arc::downgrade(v)))
                        .collect();
                    if let Ok(_) = manager.full_amount_manager.commit_weak_ptrs(weak_ptrs) {
                        for (id, entry) in embeddings.into_iter() {
                            manager.embedding_holder.insert(id, entry);
                        }

                        let cur_loaded_files = num_loaded_files.fetch_add(1, Ordering::AcqRel);
                        let cur_loaded_files = cur_loaded_files + 1;

                        let loading_progress = (cur_loaded_files as f32) / (num_total_files as f32);
                        *manager.status.write() =
                            PersiaPersistenceStatus::Loading(loading_progress);
                        tracing::debug!("load embedding progress is {}", loading_progress);

                        if num_total_files == cur_loaded_files {
                            *manager.status.write() = PersiaPersistenceStatus::Idle;
                        }
                    } else {
                        let msg = String::from("failed to commit embedding to full amount manager");
                        tracing::error!("{}", msg);
                        *manager.status.write() = PersiaPersistenceStatus::Failed(msg);
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
    ) -> Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)> {
        let embeddings: Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)> = if load_opt {
            content
                .into_iter()
                .map(|(id, entry)| (id, Arc::new(RwLock::new(entry))))
                .collect()
        } else {
            let emb: Vec<(u64, HashMapEmbeddingEntry)> = content
                .into_iter()
                .map(|(id, entry)| {
                    let emb_entry = HashMapEmbeddingEntry::from_emb(entry.emb().to_vec());
                    (id, emb_entry)
                })
                .collect();

            emb.into_iter()
                .map(|(id, entry)| (id, Arc::new(RwLock::new(entry))))
                .collect()
        };
        embeddings
    }
}
