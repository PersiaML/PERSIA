use once_cell::sync::OnceCell;
use persia_embedding_config::PersiaCommonConfig;
use persia_scheduled_thread_pool::SCHEDULED_THREAD_POOL;
use prometheus::{Encoder, HistogramOpts, Opts, TextEncoder};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

pub use prometheus::{Gauge, GaugeVec, Histogram, HistogramVec, IntCounter, IntCounterVec};

#[derive(Error, Debug, Clone)]
pub enum PersiaMetricsManagerError {
    #[error("failed to register metrics")]
    RegistryError,
    #[error("persia metrics not enabled")]
    NotEnabledError,
    #[error("persia global config not found")]
    GlobalConfigNotFoundError,
}

static PERSIA_METRICS_MANAGER: OnceCell<Arc<PersiaMetricsManager>> = OnceCell::new();

pub struct PersiaMetricsManager {
    const_labels: HashMap<String, String>,
    push_interval: std::time::Duration,
    job_name: String,
}

impl PersiaMetricsManager {
    pub fn get() -> Result<Arc<Self>, PersiaMetricsManagerError> {
        let config = PersiaCommonConfig::get();
        if config.is_err() {
            return Err(PersiaMetricsManagerError::GlobalConfigNotFoundError);
        }
        let config = config.unwrap();
        let singleton = PERSIA_METRICS_MANAGER.get_or_try_init(|| {
            let guard = config.read();
            if !guard.metrics_config.enable_metrics {
                return Err(PersiaMetricsManagerError::NotEnabledError);
            }
            let singleton = Arc::new(Self::new(
                guard.metrics_config.job_name.clone(),
                guard.metrics_config.instance_name.clone(),
                guard.metrics_config.ip_addr.clone(),
                guard.metrics_config.push_interval.clone(),
            ));
            Ok(singleton)
        });
        match singleton {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(e),
        }
    }

    fn new(job_name: String, instance_name: String, ip_addr: String, push_interval: u64) -> Self {
        let mut const_labels = HashMap::with_capacity(2);
        const_labels.insert(String::from("instance"), instance_name);
        const_labels.insert(String::from("ip_addr"), ip_addr);

        let instance = Self {
            const_labels,
            push_interval: std::time::Duration::from_secs(push_interval),
            job_name,
        };
        instance.spawn_push();
        instance
    }

    pub fn create_counter(
        &self,
        name: &str,
        help: &str,
    ) -> Result<IntCounter, PersiaMetricsManagerError> {
        let opts = Opts::new(name, help);
        let metrics = IntCounter::with_opts(opts).unwrap();
        prometheus::default_registry()
            .register(Box::new(metrics.clone()))
            .unwrap();
        Ok(metrics)
    }

    pub fn create_counter_vec(
        &self,
        name: &str,
        help: &str,
    ) -> Result<IntCounterVec, PersiaMetricsManagerError> {
        let opts = Opts::new(name, help);
        let metrics = IntCounterVec::new(opts, &["feat"]).unwrap();
        prometheus::default_registry()
            .register(Box::new(metrics.clone()))
            .unwrap();
        Ok(metrics)
    }

    pub fn create_gauge(&self, name: &str, help: &str) -> Result<Gauge, PersiaMetricsManagerError> {
        let opts = Opts::new(name, help);
        let metrics = Gauge::with_opts(opts).unwrap();
        prometheus::default_registry()
            .register(Box::new(metrics.clone()))
            .unwrap();
        Ok(metrics)
    }

    pub fn create_gauge_vec(
        &self,
        name: &str,
        help: &str,
    ) -> Result<GaugeVec, PersiaMetricsManagerError> {
        let opts = Opts::new(name, help);
        let metrics = GaugeVec::new(opts, &["feat"]).unwrap();
        prometheus::default_registry()
            .register(Box::new(metrics.clone()))
            .unwrap();
        Ok(metrics)
    }

    pub fn create_histogram(
        &self,
        name: &str,
        help: &str,
    ) -> Result<Histogram, PersiaMetricsManagerError> {
        let opts = HistogramOpts::new(name, help);
        let metrics = Histogram::with_opts(opts).unwrap();
        prometheus::default_registry()
            .register(Box::new(metrics.clone()))
            .unwrap();
        Ok(metrics)
    }

    pub fn create_histogram_vec(
        &self,
        name: &str,
        help: &str,
    ) -> Result<HistogramVec, PersiaMetricsManagerError> {
        let opts = HistogramOpts::new(name, help);
        let metrics = HistogramVec::new(opts, &["feat"]).unwrap();
        prometheus::default_registry()
            .register(Box::new(metrics.clone()))
            .unwrap();
        Ok(metrics)
    }

    fn _export_http(&self) -> () {
        let addr_raw = format!("0.0.0.0:{}", 9091);
        let binding = addr_raw.parse().unwrap();
        prometheus_exporter::start(binding).unwrap();
    }

    fn push_metrics(&self) -> () {
        let pushgateway_addr = std::env::var("PERSIA_METRICS_GATEWAY_ADDR")
            .unwrap_or(String::from("metrics_gateway:9091"));
        let metric_families = prometheus::gather();

        if let Err(e) = prometheus::push_metrics(
            self.job_name.as_str(),
            self.const_labels.clone(),
            pushgateway_addr.as_str(),
            metric_families,
            None,
        ) {
            tracing::error!("failed to push metrics to gateway, {:?}", e);
            self.log_metrics();
        } else {
            tracing::debug!("successed to push metrics");
        }
    }

    fn spawn_push(&self) -> () {
        tracing::info!("starting push metrics task...");
        SCHEDULED_THREAD_POOL.execute_at_fixed_rate(
            self.push_interval,
            self.push_interval,
            move || {
                if let Ok(metrics_manager) = Self::get() {
                    metrics_manager.push_metrics();
                } else {
                    tracing::warn!("persia metrics manager not ready");
                }
            },
        );
    }

    fn log_metrics(&self) -> () {
        let mut buffer = vec![];
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        tracing::info!("{}", String::from_utf8(buffer).unwrap());
    }
}
