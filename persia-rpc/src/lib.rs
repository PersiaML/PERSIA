use serde::de::DeserializeOwned;
use serde::Serialize;
use snafu::{ensure, Backtrace, ResultExt, Snafu};
use std::ops::Add;
pub use persia_rpc_macro::service;
use hyper::body::Buf;
use std::time::Duration;
use smol_timeout::TimeoutExt;

#[derive(Snafu, Debug)]
#[snafu(visibility = "pub")]
pub enum PersiaRpcError {
    #[snafu(display("serialization error"))]
    SerializationFailure {
        source: bincode::Error,
        backtrace: Option<Backtrace>,
    },
    #[snafu(display("timeout error"))]
    TimeoutError { },
    #[snafu(display("io error"))]
    IOFailure {
        source: std::io::Error,
        backtrace: Option<Backtrace>,
    },
    #[snafu(display("server addr parse error from {}: {}", server_addr, source))]
    ServerAddrParseFailure {
        server_addr: String,
        source: url::ParseError,
        backtrace: Option<Backtrace>,
    },
    #[snafu(display("transport error {}: {}", msg, source))]
    TransportError {
        msg: String,
        source: hyper::Error,
        backtrace: Option<Backtrace>,
    },
    #[snafu(display("transport server side error {}", msg))]
    TransportServerSideError {
        msg: String,
        backtrace: Option<Backtrace>,
    },
}

pub struct RpcClient {
    client: hyper::Client<hyper::client::HttpConnector>,
    server_addr: url::Url,
}

fn expect_uri(url: url::Url) -> hyper::Uri {
    url.as_str()
        .parse()
        .expect("a parsed Url should always be a valid Uri")
}

impl RpcClient {
    /// server_addr format should be host:port
    pub fn new(server_addr: &str) -> Result<Self, PersiaRpcError> {
        let server_addr = url::Url::parse("http://".to_string().add(server_addr).as_str())
            .context(ServerAddrParseFailure {
                server_addr: server_addr.to_string(),
            })?;
        Ok(Self {
            client: hyper::Client::builder()
                // .http2_only(true)
                // .retry_canceled_requests(true)
                // .set_host(false)
                // .http2_adaptive_window(true)
                .build_http(),
            server_addr,
        })
    }

    pub async fn call_async_timeout<T: Serialize + Send + 'static, R: DeserializeOwned + Send + 'static>(
        &self,
        endpoint_name: &str,
        input: T,
        compress: bool,
        timeout: Duration
    ) -> Result<R, PersiaRpcError> {
        let fut = self.call_async(endpoint_name, input, compress);
        let fut = fut.timeout(timeout);
        fut.await.ok_or_else(|| PersiaRpcError::TimeoutError {})?
    }

    pub async fn call_async<T: Serialize + Send + 'static, R: DeserializeOwned + Send + 'static>(
        &self,
        endpoint_name: &str,
        input: T,
        compress: bool,
    ) -> Result<R, PersiaRpcError> {
        let server_addr = self
            .server_addr
            .join(endpoint_name)
            .context(ServerAddrParseFailure {
                server_addr: endpoint_name.to_string(),
            })?;

        let data = tokio::task::block_in_place(|| bincode::serialize(&input))
            .context(SerializationFailure {})?;

        let data = if compress && data.len() > 0 {
            tokio::task::block_in_place(|| lz4::block::compress(data.as_slice(), Some(lz4::block::CompressionMode::FAST(3)), true))
                .context(IOFailure {})?
        } else {
            data
        };


        let req = hyper::Request::builder()
            .method("POST")
            .uri(expect_uri(server_addr))
            .body(hyper::Body::from(data))
            .expect("request builder");

        let response = self.client.request(req).await.context(TransportError {
            msg: format!("call {} error", endpoint_name),
        })?;
        ensure!(
            response.status() == hyper::http::StatusCode::OK,
            TransportServerSideError {
                msg: format!(
                    "call {} server side error: {:?}",
                    endpoint_name,
                    response.into_body()
                ),
            }
        );

        let resp_bytes =
            hyper::body::to_bytes(response.into_body())
                .await
                .context(TransportError {
                    msg: format!("call {} recv bytes error", endpoint_name),
                })?;

        let resp_bytes = if compress && resp_bytes.len() > 0 {
            tokio::task::block_in_place(|| lz4::block::decompress(resp_bytes.bytes(), None))
                .context(IOFailure {})?.into()
        } else {
            resp_bytes
        };

        let resp: R = tokio::task::block_in_place(|| bincode::deserialize(resp_bytes.bytes()))
            .context(SerializationFailure {})?;
        Ok(resp)
    }
}
