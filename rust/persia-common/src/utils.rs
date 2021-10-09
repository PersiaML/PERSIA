use persia_libs::{flume, parking_lot, tracing};

#[derive(Clone)]
pub struct ChannelPair<T> {
    pub sender: flume::Sender<T>,
    pub receiver: flume::Receiver<T>,
}

impl<T> ChannelPair<T> {
    pub fn new(cap: usize) -> Self {
        let (sender, receiver) = flume::bounded(cap);
        Self { sender, receiver }
    }

    pub fn new_unbounded() -> Self {
        let (sender, receiver) = flume::unbounded();
        Self { sender, receiver }
    }
}

pub fn start_deadlock_detection_thread() {
    if std::env::var("PERSIA_DEADLOCK_DETECTION")
        .unwrap_or(String::from("0"))
        .parse::<usize>()
        .expect("PERSIA_DEADLOCK_DETECTION should be 0 or 1")
        > 0
    {
        std::thread::spawn(move || {
            tracing::info!("deadlock detection thread started");
            loop {
                std::thread::sleep(std::time::Duration::from_secs(60));
                let deadlocks = parking_lot::deadlock::check_deadlock();
                if deadlocks.is_empty() {
                    continue;
                }

                tracing::error!("{} deadlocks detected", deadlocks.len());
                for (i, threads) in deadlocks.iter().enumerate() {
                    tracing::error!("Deadlock #{}", i);
                    for t in threads {
                        tracing::error!("Thread Id {:#?}", t.thread_id());
                        tracing::error!("{:#?}", t.backtrace());
                    }
                }
            }
        });
    }
}
