use persia_metrics::{Gauge, IntCounter, PersiaMetricsManager, PersiaMetricsManagerError};

use persia_libs::{anyhow::Result, once_cell::sync::OnceCell};

static METRICS_HOLDER: OnceCell<MetricsHolder> = OnceCell::new();

pub struct MetricsHolder {
    pub forward_client_to_gpu_time_cost_ms: Gauge,
    pub forward_client_time_cost_ms: Gauge,
    pub forward_error: IntCounter,
    pub backward_client_time_cost_ms: Gauge,
    pub long_get_train_batch_time_cost_ms: Gauge,
    pub long_update_gradient_batched_time_cost_ms: Gauge,
}

impl MetricsHolder {
    pub fn get() -> Result<&'static Self, PersiaMetricsManagerError> {
        METRICS_HOLDER.get_or_try_init(|| {
            let m = PersiaMetricsManager::get()?;
            let holder = Self {
                forward_client_to_gpu_time_cost_ms: m.create_gauge(
                    "forward_client_to_gpu_time_cost_ms",
                    "get batched dense data and embeddings, then send to device time cost"
                )?,
                forward_client_time_cost_ms: m.create_gauge("forward_client_time_cost_ms", "get embeddings time cost")?,
                forward_error: m.create_counter("forward_error", "get embedding error counter")?,
                backward_client_time_cost_ms: m.create_gauge(
                    "backward_client_time_cost_ms", 
                    "get graident of sparse feature and update it to server time cost"
                )?,
                long_get_train_batch_time_cost_ms: m.create_gauge(
                    "long_get_train_batch_time_cost_ms",
                    "get train batch time cost when it takes more than 1ms"
                )?,
                long_update_gradient_batched_time_cost_ms: m.create_gauge(
                    "long_update_gradient_batched_time_cost_ms",
                    "send gradient of sparse feature to gradient update buffer time cost when it takes more than 1ms"
                )?,
            };
            Ok(holder)
        })
    }
}
