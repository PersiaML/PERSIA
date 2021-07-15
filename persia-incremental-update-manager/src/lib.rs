#![allow(clippy::needless_return)]

use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use griddle::HashMap;
use once_cell::sync::OnceCell;
use parking_lot::{Mutex, RwLock};
use rayon::{ThreadPool, ThreadPoolBuilder};
use thiserror::Error;

use persia_embedding_config::{
    PerisaIntent, PersiaCommonConfig, PersiaGlobalConfigError, PersiaPersistenceStorage,
    PersiaReplicaInfo, PersiaShardedServerConfig,
};
use persia_embedding_datatypes::HashMapEmbeddingEntry;
use persia_embedding_holder::{PersiaEmbeddingHolder, PersiaEmbeddingHolderError};
use persia_futures::ChannelPair;
use persia_metrics::{Gauge, PersiaMetricsManager, PersiaMetricsManagerError};
use persia_storage_visitor::{
    PerisaIncrementalPacket, PersiaCephVisitor, PersiaHdfsVisitor, PersiaStorageVisitor, SpeedyObj,
};

#[derive(Error, Debug)]
pub enum IncrementalUpdateError {
    #[error("embedding holder error: {0}")]
    PersiaEmbeddingHolderError(#[from] PersiaEmbeddingHolderError),
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("embedding holder not found error")]
    CommitIncrementalError,
}

static METRICS_HOLDER: OnceCell<MetricsHolder> = OnceCell::new();

struct MetricsHolder {
    pub inc_update_delay: Gauge,
}

impl MetricsHolder {
    pub fn get() -> Result<&'static Self, PersiaMetricsManagerError> {
        METRICS_HOLDER.get_or_try_init(|| {
            let m = PersiaMetricsManager::get()?;
            let holder = Self {
                inc_update_delay: m.create_gauge("inc_update_delay", "ATT")?,
            };
            Ok(holder)
        })
    }
}

pub fn current_unix_time() -> u64 {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs() as u64;
    return since_the_epoch;
}

static INCREMENTAL_UPDATE_MANAGER: OnceCell<Arc<PerisaIncrementalUpdateManager>> = OnceCell::new();

pub struct PerisaIncrementalUpdateManager {
    storage_visitor: Arc<dyn PersiaStorageVisitor>,
    embedding_holder: PersiaEmbeddingHolder,
    executors: Arc<ThreadPool>,
    sign_per_file: usize,
    shard_idx: usize,
    incremental_buffer_size: usize,
    incremental_dir: std::path::PathBuf,
    buffer_channel_input: ChannelPair<Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)>>,
    buffer_channel_output: ChannelPair<Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)>>,
    _background_threads: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>,
}

impl PerisaIncrementalUpdateManager {
    pub fn get() -> Result<Arc<Self>, IncrementalUpdateError> {
        let singleton = INCREMENTAL_UPDATE_MANAGER.get_or_try_init(|| {
            let server_config = PersiaShardedServerConfig::get()?;
            let common_comfig = PersiaCommonConfig::get()?;
            let embedding_holder = PersiaEmbeddingHolder::get()?;
            let replica_info = PersiaReplicaInfo::get()?;

            let singleton = Self::new(
                server_config.storage.clone(),
                embedding_holder,
                common_comfig.intent.clone(),
                server_config.num_persistence_workers,
                server_config.num_signs_per_file,
                replica_info.replica_index,
                server_config.incremental_buffer_size,
                server_config.incremental_dir.clone(),
                server_config.incremental_channel_capacity,
            );

            Ok(singleton)
        });
        match singleton {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(e),
        }
    }

    fn new(
        storage: PersiaPersistenceStorage,
        embedding_holder: PersiaEmbeddingHolder,
        cur_task: PerisaIntent,
        num_executors: usize,
        sign_per_file: usize,
        shard_idx: usize,
        incremental_buffer_size: usize,
        incremental_dir: String,
        update_channel_capacity: usize,
    ) -> Arc<Self> {
        let executors = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(num_executors)
                .build()
                .unwrap(),
        );
        let storage_visitor: Arc<dyn PersiaStorageVisitor> = match storage {
            PersiaPersistenceStorage::Ceph => Arc::new(PersiaCephVisitor {}),
            PersiaPersistenceStorage::Hdfs => Arc::new(PersiaHdfsVisitor::new()),
        };
        let buffer_channel_input: ChannelPair<Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)>> =
            ChannelPair::new(update_channel_capacity);

        let buffer_channel_output: ChannelPair<Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)>> =
            ChannelPair::new(update_channel_capacity);

        let incremental_dir = [incremental_dir, format!("s{}", shard_idx)]
            .iter()
            .collect();
        let background_threads = Arc::new(Mutex::new(vec![]));

        let instance = Arc::new(Self {
            storage_visitor,
            embedding_holder,
            executors,
            sign_per_file,
            shard_idx,
            incremental_buffer_size,
            incremental_dir,
            buffer_channel_input,
            buffer_channel_output,
            _background_threads: background_threads.clone(),
        });

        let mut handle_guard = background_threads.lock();
        match cur_task {
            PerisaIntent::Train => {
                let handle = std::thread::spawn({
                    let instance = instance.clone();
                    move || {
                        instance.buffer_input_thread();
                    }
                });
                handle_guard.push(handle);

                let handle = std::thread::spawn({
                    let instance = instance.clone();
                    move || {
                        instance.buffer_output_thread();
                    }
                });
                handle_guard.push(handle);
            }
            PerisaIntent::Infer(_) => {
                let handle = std::thread::spawn({
                    let instance = instance.clone();
                    move || {
                        instance.inc_dir_scan_thread();
                    }
                });
                handle_guard.push(handle);
            }
            _ => {}
        }

        instance
    }

    fn dump_embedding_segment(
        &self,
        dst_dir: PathBuf,
        segment: Vec<(u64, Vec<f32>)>,
        file_index: usize,
        num_dumped_signs: Arc<AtomicUsize>,
        num_total_signs: usize,
    ) -> () {
        let segment_len = segment.len();
        let file_name = PathBuf::from(format!("{}_{}.inc", self.shard_idx, file_index));

        let content = SpeedyObj::PerisaIncrementalPacket(PerisaIncrementalPacket {
            content: segment,
            timestamps: current_unix_time(),
        });

        if let Err(e) =
            self.storage_visitor
                .dump_to_file_speedy(content, dst_dir.clone(), file_name.clone())
        {
            tracing::error!(
                "failed to dump {:?} inc update packet to {:?}, because {:?}",
                file_name,
                dst_dir,
                e
            );
        } else {
            let dumped = num_dumped_signs.fetch_add(segment_len, Ordering::AcqRel);
            let cur_dumped = dumped + segment_len;
            if cur_dumped >= num_total_signs {
                let done_file = PathBuf::from("inc_done");
                if let Err(e) = self.storage_visitor.create_file(dst_dir, done_file) {
                    tracing::error!("failed to mark increment update done, {:?}", e);
                }
            }
        }
    }

    fn load_embedding_from_file(&self, file_path: PathBuf) -> () {
        if let Ok(speedy_content) = self.storage_visitor.read_from_file_speedy(file_path) {
            match speedy_content {
                SpeedyObj::PerisaIncrementalPacket(packet) => {
                    let delay = current_unix_time() - packet.timestamps;
                    if let Ok(m) = MetricsHolder::get() {
                        m.inc_update_delay.set(delay as f64);
                    }
                    tracing::debug!("loading inc packet, delay is {}s", delay);
                    packet.content.into_iter().for_each(|(id, emb)| {
                        self.embedding_holder.inner.insert(
                            id,
                            Arc::new(RwLock::new(HashMapEmbeddingEntry::from_emb_infer(emb))),
                        );
                    });
                }
                _ => {}
            }
        }
    }

    pub fn try_commit_incremental(
        &self,
        incremental: Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)>,
    ) -> Result<(), IncrementalUpdateError> {
        let res = self.buffer_channel_input.sender.try_send(incremental);
        if res.is_err() {
            Err(IncrementalUpdateError::CommitIncrementalError)
        } else {
            Ok(())
        }
    }

    fn buffer_input_thread(&self) -> () {
        let mut sending_buffer = HashMap::with_capacity(self.incremental_buffer_size);
        self.buffer_channel_input
            .receiver
            .iter()
            .for_each(|emb_vec| {
                for (k, v) in emb_vec.into_iter() {
                    sending_buffer.insert(k, v);
                }
                if sending_buffer.len() > self.incremental_buffer_size {
                    let mut indices = Vec::with_capacity(sending_buffer.len());
                    for (k, v) in sending_buffer.iter() {
                        indices.push((k.clone(), v.clone()));
                    }
                    sending_buffer.clear();
                    if let Err(_) = self.buffer_channel_output.sender.try_send(indices) {
                        tracing::warn!("failed to inc update, please try a bigger inc buffer size");
                    }
                }
            })
    }

    fn buffer_output_thread(&self) -> () {
        self.buffer_channel_output
            .receiver
            .iter()
            .for_each(|embeddings| {
                let num_total_signs = embeddings.len();
                let num_dumped_signs = Arc::new(AtomicUsize::new(0));
                for (file_index, segment) in embeddings.chunks(self.sign_per_file).enumerate() {
                    let emb_without_opt = segment
                        .iter()
                        .map(|x| (x.0, x.1.read().emb().to_vec()))
                        .collect();

                    let inc_dir_name =
                        PathBuf::from(chrono::Local::now().format("inc_%Y%m%d%H%M%S").to_string());
                    let cur_inc_dir: PathBuf = [self.incremental_dir.clone(), inc_dir_name]
                        .iter()
                        .collect();

                    if let Ok(manager) = Self::get() {
                        manager.executors.spawn({
                            let manager = manager.clone();
                            let num_dumped_signs = num_dumped_signs.clone();
                            move || {
                                manager.dump_embedding_segment(
                                    cur_inc_dir,
                                    emb_without_opt,
                                    file_index,
                                    num_dumped_signs,
                                    num_total_signs,
                                );
                            }
                        });
                    }
                }
            });
    }

    fn inc_dir_scan_thread(&self) -> () {
        let inc_dir = self.incremental_dir.clone();
        if !inc_dir.is_dir() {
            tracing::error!("incremental_dir is not a dir");
            return;
        }
        tracing::info!("start to scan dir {:?}", inc_dir);
        let mut dir_set = std::collections::HashSet::new();

        if let Ok(cur_inc_dirs) = self.storage_visitor.list_dir(inc_dir.clone()) {
            cur_inc_dirs.into_iter().for_each(|d| {
                if d.is_dir() {
                    dir_set.insert(d);
                }
            });
        }

        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            if let Ok(cur_inc_dirs) = self.storage_visitor.list_dir(inc_dir.clone()) {
                cur_inc_dirs.into_iter().for_each(|d| {
                    if !dir_set.contains(&d) && d.is_dir() {
                        let done_file = [d.clone(), PathBuf::from("inc_done")].iter().collect();
                        if self.storage_visitor.is_file(done_file).unwrap_or(false) {
                            if let Ok(file_list) = self.storage_visitor.list_dir(d.clone()) {
                                let file_list: Vec<PathBuf> = file_list
                                    .into_iter()
                                    .filter(|x| x.extension() == Some(OsStr::new("inc")))
                                    .collect();
                                file_list.into_iter().for_each(|f| {
                                    if let Ok(manager) = Self::get() {
                                        manager.executors.spawn({
                                            let manager = manager.clone();
                                            move || {
                                                manager.load_embedding_from_file(f);
                                            }
                                        });
                                    }
                                });
                            }
                        }
                        dir_set.insert(d);
                    }
                });
            }
        }
    }
}
