#![allow(clippy::needless_return)]

use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use griddle::HashSet;
use persia_libs::{
    chrono,
    itertools::Itertools,
    once_cell::sync::OnceCell,
    rayon::{ThreadPool, ThreadPoolBuilder},
    thiserror, tracing,
};

use persia_common::utils::ChannelPair;
use persia_embedding_config::{
    EmbeddingParameterServerConfig, PerisaJobType, PersiaCommonConfig, PersiaGlobalConfigError,
    PersiaReplicaInfo,
};
use persia_embedding_holder::{
    emb_entry::HashMapEmbeddingEntry, PersiaEmbeddingHolder, PersiaEmbeddingHolderError,
};
use persia_metrics::{Gauge, PersiaMetricsManager, PersiaMetricsManagerError};
use persia_speedy::{Readable, Writable};
use persia_storage::{PersiaPath, PersiaPathImpl};

#[derive(Readable, Writable, Debug)]
pub struct PerisaIncrementalPacket {
    pub content: Vec<HashMapEmbeddingEntry>,
    pub timestamps: u64,
}

#[derive(thiserror::Error, Debug)]
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
    pub inc_update_delay_sec: Gauge,
}

impl MetricsHolder {
    pub fn get() -> Result<&'static Self, PersiaMetricsManagerError> {
        METRICS_HOLDER.get_or_try_init(|| {
            let m = PersiaMetricsManager::get()?;
            let holder = Self {
                inc_update_delay_sec: m.create_gauge(
                    "inc_update_delay_sec",
                    "The time delay between the package being dumped and being loaded",
                )?,
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
const INCREMENTAL_UPDATE_CHANNEL_CAPACITY: usize = 1000;

pub struct PerisaIncrementalUpdateManager {
    embedding_holder: PersiaEmbeddingHolder,
    executors: Arc<ThreadPool>,
    replica_index: usize,
    incremental_buffer_size: usize,
    incremental_dir: std::path::PathBuf,
    buffer_channel_input: ChannelPair<Vec<u64>>,
    buffer_channel_output: ChannelPair<Vec<u64>>,
}

impl PerisaIncrementalUpdateManager {
    pub fn get() -> Result<Arc<Self>, IncrementalUpdateError> {
        let singleton = INCREMENTAL_UPDATE_MANAGER.get_or_try_init(|| {
            let server_config = EmbeddingParameterServerConfig::get()?;
            let common_config = PersiaCommonConfig::get()?;
            let embedding_holder = PersiaEmbeddingHolder::get()?;
            let replica_info = PersiaReplicaInfo::get()?;

            let singleton = Self::new(
                embedding_holder,
                common_config.job_type.clone(),
                common_config.checkpointing_config.num_workers,
                replica_info.replica_index,
                server_config.incremental_buffer_size,
                server_config.incremental_dir.clone(),
            );

            Ok(singleton)
        });
        match singleton {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(e),
        }
    }

    fn new(
        embedding_holder: PersiaEmbeddingHolder,
        cur_task: PerisaJobType,
        num_executors: usize,
        replica_index: usize,
        incremental_buffer_size: usize,
        incremental_dir: String,
    ) -> Arc<Self> {
        let executors = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(num_executors)
                .build()
                .unwrap(),
        );
        let buffer_channel_input: ChannelPair<Vec<u64>> =
            ChannelPair::new(INCREMENTAL_UPDATE_CHANNEL_CAPACITY);

        let buffer_channel_output: ChannelPair<Vec<u64>> =
            ChannelPair::new(INCREMENTAL_UPDATE_CHANNEL_CAPACITY);

        let incremental_dir = [incremental_dir, format!("s{}", replica_index)]
            .iter()
            .collect();

        let instance = Arc::new(Self {
            embedding_holder,
            executors,
            replica_index,
            incremental_buffer_size,
            incremental_dir,
            buffer_channel_input,
            buffer_channel_output,
        });

        match cur_task {
            PerisaJobType::Train => {
                std::thread::spawn({
                    let instance = instance.clone();
                    move || {
                        instance.buffer_input_thread();
                    }
                });

                std::thread::spawn({
                    let instance = instance.clone();
                    move || {
                        instance.buffer_output_thread();
                    }
                });
            }
            PerisaJobType::Infer => {
                std::thread::spawn({
                    let instance = instance.clone();
                    move || {
                        instance.inc_dir_scan_thread();
                    }
                });
            }
            _ => {}
        }

        instance
    }

    fn dump_embedding_segment(
        &self,
        dst_dir: PathBuf,
        signs: Vec<u64>,
        file_index: usize,
        num_dumped_signs: Arc<AtomicUsize>,
        num_total_signs: usize,
    ) -> () {
        let mut entries = Vec::with_capacity(signs.len());
        signs.iter().for_each(|sign| {
            let shard = self.embedding_holder.shard(sign).read();
            if let Some(entry) = shard.get(sign) {
                entries.push(entry.clone());
            }
        });

        let segment_len = entries.len();
        let file_name = PathBuf::from(format!("{}_{}.inc", self.replica_index, file_index));

        let content = PerisaIncrementalPacket {
            content: entries,
            timestamps: current_unix_time(),
        };

        let emb_path = PersiaPath::from_vec(vec![&dst_dir, &file_name]);
        if let Err(e) = emb_path.write_all_speedy(&content) {
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
                let inc_update_done_file = PathBuf::from("inc_update_done");
                let inc_update_done_path =
                    PersiaPath::from_vec(vec![&dst_dir, &inc_update_done_file]);
                if let Err(e) = inc_update_done_path.create(false) {
                    tracing::error!("failed to mark increment update done, {:?}", e);
                }
            }
        }
    }

    fn load_embedding_from_file(&self, file_path: PathBuf) -> () {
        let file_path = PersiaPath::from_pathbuf(file_path);
        let packet: PerisaIncrementalPacket = file_path.read_to_end_speedy().unwrap();
        let delay = current_unix_time() - packet.timestamps;
        if let Ok(m) = MetricsHolder::get() {
            m.inc_update_delay_sec.set(delay as f64);
        }
        tracing::debug!("loading inc packet, delay is {}s", delay);
        packet.content.into_iter().for_each(|entry| {
            let sign = entry.sign();
            let mut shard = self.embedding_holder.shard(&sign).write();
            shard.insert(sign, entry);
        });
    }

    pub fn try_commit_incremental(
        &self,
        incremental: Vec<u64>,
    ) -> Result<(), IncrementalUpdateError> {
        let res = self.buffer_channel_input.sender.try_send(incremental);
        if res.is_err() {
            Err(IncrementalUpdateError::CommitIncrementalError)
        } else {
            Ok(())
        }
    }

    fn buffer_input_thread(&self) -> () {
        let mut sending_buffer = HashSet::with_capacity(self.incremental_buffer_size);
        self.buffer_channel_input
            .receiver
            .iter()
            .for_each(|emb_vec| {
                emb_vec.into_iter().for_each(|sign| {
                    sending_buffer.insert(sign);
                });

                if sending_buffer.len() > self.incremental_buffer_size {
                    let indices = sending_buffer.iter().copied().collect_vec();
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
            .for_each(|signs| {
                let num_total_signs = signs.len();
                let num_dumped_signs = Arc::new(AtomicUsize::new(0));
                let sign_per_file = num_total_signs / self.executors.current_num_threads();

                let chunk_signs: Vec<Vec<u64>> = signs
                    .into_iter()
                    .chunks(sign_per_file)
                    .into_iter()
                    .map(|chunk| chunk.collect())
                    .collect();

                for (file_index, signs_slice) in chunk_signs.into_iter().enumerate() {
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
                                    signs_slice,
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
            tracing::warn!("incremental_dir is not a dir");
            return;
        }
        tracing::info!("start to scan dir {:?}", inc_dir);
        let mut dir_set = std::collections::HashSet::new();

        let inc_dir = PersiaPath::from_pathbuf(inc_dir);

        if let Ok(cur_inc_dirs) = inc_dir.list() {
            cur_inc_dirs.into_iter().for_each(|d| {
                if d.is_dir() {
                    dir_set.insert(d);
                }
            });
        }

        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            if let Ok(cur_inc_dirs) = inc_dir.list() {
                cur_inc_dirs.into_iter().for_each(|d| {
                    if !dir_set.contains(&d) && d.is_dir() {
                        let inc_update_done = PathBuf::from("inc_update_done");
                        let inc_update_done_file = PersiaPath::from_vec(vec![&d, &inc_update_done]);
                        if inc_update_done_file.is_file().unwrap_or(false) {
                            let inc_dir = PersiaPath::from_pathbuf(d.clone());
                            if let Ok(file_list) = inc_dir.list() {
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
