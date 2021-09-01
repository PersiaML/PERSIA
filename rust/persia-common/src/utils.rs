use persia_libs::flume;

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
