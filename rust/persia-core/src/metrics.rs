use persia_metrics::{Histogram, IntCounter, PersiaMetricsManager, PersiaMetricsManagerError};

use persia_libs::{anyhow::Result, once_cell::sync::OnceCell};

static METRICS_HOLDER: OnceCell<MetricsHolder> = OnceCell::new();

pub struct MetricsHolder {
    pub forward_client_to_gpu_time_cost: Histogram,
    pub forward_client_time_cost: Histogram,
    pub forward_error: IntCounter,
    pub backward_client_time_cost: Histogram,
    pub long_get_train_batch_time_cost: Histogram,
    pub long_update_gradient_batched_time_cost: Histogram,
}

impl MetricsHolder {
    pub fn get() -> Result<&'static Self, PersiaMetricsManagerError> {
        METRICS_HOLDER.get_or_try_init(|| {
            let m = PersiaMetricsManager::get()?;
            let holder = Self {
                forward_client_to_gpu_time_cost: m
                    .create_histogram("forward_client_to_gpu_time_cost", "ATT")?,
                forward_client_time_cost: m.create_histogram("forward_client_time_cost", "ATT")?,
                forward_error: m.create_counter("forward_error", "ATT")?,
                backward_client_time_cost: m
                    .create_histogram("backward_client_time_cost", "ATT")?,
                long_get_train_batch_time_cost: m
                    .create_histogram("long_get_train_batch_time_cost", "ATT")?,
                long_update_gradient_batched_time_cost: m
                    .create_histogram("long_update_gradient_batched_time_cost", "ATT")?,
            };
            Ok(holder)
        })
    }
}
