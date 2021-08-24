pub use async_channel;
pub use async_compat;
pub use async_executor;
pub use async_lock;
pub use async_oneshot;
pub use easy_parallel;
pub use flume;
pub use futures;
pub use smol;
pub use smol_timeout;
pub use tokio;

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
