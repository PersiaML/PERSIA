use once_cell::sync::Lazy;
use scheduled_thread_pool::ScheduledThreadPool;

#[allow(dead_code)]
pub static SCHEDULED_THREAD_POOL: Lazy<ScheduledThreadPool> = Lazy::new(|| {
    ScheduledThreadPool::with_name(
        "persia_scheduled_thread_pool",
        std::env::var("PERSIA_SCHEDULED_THREAD_POOL_SIZE")
            .unwrap_or_else(|_| "1".into())
            .parse()
            .expect("set scheduled thread pool size error"),
    )
});
