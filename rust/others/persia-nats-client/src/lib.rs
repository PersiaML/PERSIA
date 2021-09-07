use std::time::Duration;

use async_nats::{Connection, Subscription};
use persia_libs::{
    once_cell::sync::OnceCell,
    retry::{delay::Fixed, retry},
    smol, thiserror, tracing,
};

use persia_speedy::{Readable, Writable};

#[derive(Readable, Writable, thiserror::Error, Clone, Debug)]
pub enum NatsError {
    #[error("nats io error {0:?}")]
    IoError(String),
    #[error("decode speedy error")]
    DecodeError,
    #[error("not found any node in subject error")]
    EmptyNodeError,
}

impl From<std::io::Error> for NatsError {
    fn from(error: std::io::Error) -> Self {
        let msg = format!("{:?}", error);
        NatsError::IoError(msg)
    }
}

static NATS_CLIRNT: OnceCell<NatsClient> = OnceCell::new();

#[derive(Debug, Clone)]
pub struct NatsClient {
    nc: Connection,
    timeout: Duration,
}

impl NatsClient {
    pub fn get() -> Self {
        NATS_CLIRNT.get_or_init(|| NatsClient::new()).clone()
    }
    fn new() -> Self {
        let nats_url = std::env::var("PERSIA_NATS_IP")
            .unwrap_or(String::from("nats://persia_nats_service:4222"));
        let nc = retry(Fixed::from_millis(5000), || {
            let res = smol::block_on(async_nats::connect(nats_url.as_str()));
            if res.is_err() {
                tracing::warn!("failed to connect nats server, {:?}", res);
            }
            res
        })
        .expect("failed to init nats connection");

        Self {
            nc,
            timeout: Duration::from_secs(10),
        }
    }

    pub async fn subscribe(&self, subject: &str) -> Result<Subscription, NatsError> {
        match self.nc.subscribe(subject).await {
            Ok(subscription) => Ok(subscription),
            Err(err) => Err(NatsError::from(err)),
        }
    }

    pub async fn request(&self, subject: &str, msg: &[u8]) -> Result<Vec<u8>, NatsError> {
        match self.nc.request_timeout(subject, msg, self.timeout).await {
            Ok(msg) => Ok(msg.data),
            Err(err) => Err(NatsError::from(err)),
        }
    }

    pub async fn request_multi(
        &self,
        subject: &str,
        msg: &[u8],
    ) -> Result<Vec<Vec<u8>>, NatsError> {
        match self.nc.request_multi(subject, msg).await {
            Ok(subscription) => {
                let mut messages = Vec::new();
                while let Some(msg) = subscription.next().await {
                    messages.push(msg.data);
                }
                Ok(messages)
            }
            Err(err) => Err(NatsError::from(err)),
        }
    }

    pub fn get_subject(
        &self,
        service_type: &str,
        fn_name: &str,
        replica_index: Option<usize>,
    ) -> String {
        match replica_index {
            Some(idx) => format!("{}.{}.{}", service_type, fn_name, idx),
            None => format!("{}.{}", service_type, fn_name),
        }
    }
}
