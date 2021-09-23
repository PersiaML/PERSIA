use persia_libs::bytes::Buf;
use persia_libs::{hyper, lz4, tokio, url};
use persia_speedy::{Readable, Writable};
use snafu::{ensure, Backtrace, ResultExt, Snafu};
use std::ops::Add;

#[derive(Snafu, Debug)]
#[snafu(visibility = "pub")]
pub enum PersiaRpcError {
    #[snafu(display("serialization error"))]
    SerializationFailure {
        source: persia_speedy::Error,
        backtrace: Option<Backtrace>,
    },
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

    pub async fn call_async<'a, T, R>(
        &self,
        endpoint_name: &str,
        input: &T,
        compress: bool,
    ) -> Result<R, PersiaRpcError>
    where
        R: Readable<'a, persia_speedy::LittleEndian> + Send + 'static,
        T: Writable<persia_speedy::LittleEndian> + Send + 'static,
    {
        let server_addr = self
            .server_addr
            .join(endpoint_name)
            .context(ServerAddrParseFailure {
                server_addr: endpoint_name.to_string(),
            })?;

        let data = tokio::task::block_in_place(|| input.write_to_vec())
            .context(SerializationFailure {})?;

        let data = if compress && (data.len() > 0) {
            tokio::task::block_in_place(|| {
                lz4::block::compress(
                    data.as_slice(),
                    Some(lz4::block::CompressionMode::FAST(3)),
                    true,
                )
            })
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

        let mut resp_bytes =
            hyper::body::aggregate(response.into_body())
                .await
                .context(TransportError {
                    msg: format!("call {} recv bytes error", endpoint_name),
                })?;

        if compress && resp_bytes.remaining() >= 4 {
            let resp_bytes = tokio::task::block_in_place(|| {
                let mut buffer = vec![0; resp_bytes.remaining()];
                resp_bytes.copy_to_slice(buffer.as_mut());
                lz4::block::decompress(buffer.as_slice(), None)
            })
            .context(IOFailure {})?;
            let resp: R =
                tokio::task::block_in_place(|| R::read_from_buffer_owned(resp_bytes.as_slice())) // TODO: this can be zero copy if we use read_from_buffer and correctly deal with lifetime
                    .context(SerializationFailure {})?;
            return Ok(resp);
        } else {
            let resp: R =
                tokio::task::block_in_place(|| R::read_from_stream_unbuffered(resp_bytes.reader())) // TODO: this can be zero copy if we use read_from_buffer and correctly deal with lifetime
                    .context(SerializationFailure {})?;
            return Ok(resp);
        }
    }
}
