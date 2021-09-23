use std::sync::Arc;

use persia_libs::hyperloglogplus::HyperLogLog;
use persia_libs::{hashbrown, hyperloglogplus, once_cell, parking_lot, tracing};

use persia_common::{utils::ChannelPair, SingleSignInFeatureBatch};
use persia_metrics::{GaugeVec, PersiaMetricsManager, PersiaMetricsManagerError};

const INDICES_CHANNEL_CAP: usize = 1000;

static METRICS_HOLDER: once_cell::sync::OnceCell<MetricsHolder> = once_cell::sync::OnceCell::new();

struct MetricsHolder {
    pub estimated_distinct_id: GaugeVec,
}

impl MetricsHolder {
    pub fn get() -> Result<&'static Self, PersiaMetricsManagerError> {
        METRICS_HOLDER.get_or_try_init(|| {
            let m = PersiaMetricsManager::get()?;
            let holder = Self {
                estimated_distinct_id: m.create_gauge_vec("estimated_distinct_id", "ATT")?,
            };
            Ok(holder)
        })
    }
}

pub struct EmbeddingMonitorInner {
    _feature_name: String,
    distinct_id_estimator: Arc<
        parking_lot::Mutex<
            hyperloglogplus::HyperLogLogPlus<u64, hashbrown::hash_map::DefaultHashBuilder>,
        >,
    >,
    indices_channel: ChannelPair<Vec<u64>>,
    _handlers: Vec<std::thread::JoinHandle<()>>,
}

impl EmbeddingMonitorInner {
    pub fn new(feature_name: String) -> Self {
        let indices_channel = ChannelPair::new(INDICES_CHANNEL_CAP);
        let distinct_id_estimator = Arc::new(parking_lot::Mutex::new(
            hyperloglogplus::HyperLogLogPlus::new(
                16,
                hashbrown::hash_map::DefaultHashBuilder::default(),
            )
            .unwrap(),
        ));
        let mut handlers = Vec::new();

        let recv_handler = {
            let reveiver = indices_channel.receiver.clone();
            let feature_name = feature_name.clone();
            let distinct_id_estimator = distinct_id_estimator.clone();
            std::thread::spawn(move || {
                tracing::info!(
                    "background thread for estimating {} distinct id start...",
                    feature_name
                );
                loop {
                    let indices = reveiver.recv().unwrap_or(vec![]);
                    let mut estimator = distinct_id_estimator.lock();
                    indices.iter().for_each(|id| {
                        estimator.insert(id);
                    })
                }
            })
        };
        handlers.push(recv_handler);

        let commit_handler = {
            let distinct_id_estimator = distinct_id_estimator.clone();
            let feature_name = feature_name.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                if let Ok(m) = MetricsHolder::get() {
                    let distinct_id = { distinct_id_estimator.lock().count().trunc() as u64 };
                    tracing::debug!("distinct_id for {} is {}", feature_name, distinct_id);
                    m.estimated_distinct_id
                        .with_label_values(&[feature_name.as_str()])
                        .set(distinct_id as f64);
                }
            })
        };
        handlers.push(commit_handler);

        EmbeddingMonitorInner {
            _feature_name: feature_name,
            distinct_id_estimator,
            indices_channel,
            _handlers: handlers,
        }
    }
}

impl EmbeddingMonitorInner {
    pub fn monitor_index_batch(&self, index_batch: &Vec<SingleSignInFeatureBatch>) {
        let channel_size = self.indices_channel.sender.len();
        if channel_size > INDICES_CHANNEL_CAP {
            tracing::warn!("too many batches when estimating distinct id, skiping...");
            return;
        }
        let indices: Vec<u64> = index_batch.iter().map(|x| x.sign.clone()).collect();
        let result = self.indices_channel.sender.try_send(indices);
        if result.is_err() {
            tracing::warn!("too many batches when estimating distinct id, skiping...");
        }
    }

    pub fn estimate_distinct_id(&self) -> usize {
        self.distinct_id_estimator.lock().count().trunc() as usize
    }
}
