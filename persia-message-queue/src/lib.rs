use std::ops::Add;
use std::sync::Arc;

use hyper::{self, Body, Request, Response};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PersiaMessageQueueError {
    #[error("send error")]
    SendError,
    #[error("recv error")]
    RecvError,
    #[error("hyper error")]
    HyperError(#[from] hyper::Error),
}

#[derive(Clone)]
pub struct PersiaMessageQueueClient {
    client: hyper::Client<hyper::client::HttpConnector>,
    server_addr: url::Url,
}

fn expect_uri(url: url::Url) -> hyper::Uri {
    url.as_str()
        .parse()
        .expect("a parsed Url should always be a valid Uri")
}

impl PersiaMessageQueueClient {
    pub fn new(server_addr: &str) -> Self {
        let server_addr = url::Url::parse("http://".to_string().add(server_addr).as_str()).unwrap();
        Self {
            client: hyper::Client::builder()
                .http2_only(true)
                .retry_canceled_requests(true)
                .set_host(false)
                .http2_adaptive_window(true)
                .build_http(),
            server_addr,
        }
    }

    pub async fn send(&self, content: Vec<u8>) -> Result<(), PersiaMessageQueueError> {
        let req = hyper::Request::builder()
            .method("POST")
            .uri(expect_uri(self.server_addr.join("send").unwrap()))
            .body(hyper::Body::from(content))
            .expect("request builder");
        let resp = self.client.request(req).await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(PersiaMessageQueueError::SendError)
        }
    }

    pub async fn recv(&self) -> Result<Vec<u8>, PersiaMessageQueueError> {
        let req = hyper::Request::builder()
            .method("POST")
            .uri(expect_uri(self.server_addr.join("recv").unwrap()))
            .body(hyper::Body::empty())
            .expect("request builder");
        let resp = self.client.request(req).await?;
        if resp.status().is_success() {
            Ok(hyper::body::to_bytes(resp.into_body()).await?.to_vec())
        } else {
            Err(PersiaMessageQueueError::SendError)
        }
    }
}

#[derive(Clone)]
pub struct PersiaMessageQueueService {
    message_queue: persia_futures::ChannelPair<hyper::body::Bytes>,
}

#[derive(Clone)]
pub struct PersiaMessageQueueServer {
    message_queue: persia_futures::ChannelPair<hyper::body::Bytes>,
    server_handler: Arc<persia_futures::tokio::task::JoinHandle<hyper::Result<()>>>,
}

impl PersiaMessageQueueServer {
    pub fn new(port: u16, cap: usize) -> PersiaMessageQueueServer {
        let message_queue = persia_futures::ChannelPair::new(cap);
        let service = PersiaMessageQueueService {
            message_queue: message_queue.clone(),
        };

        let server = hyper::Server::bind(&([0, 0, 0, 0], port).into())
            .http2_only(true)
            .http2_adaptive_window(true)
            .tcp_nodelay(true)
            .serve(hyper::service::make_service_fn(move |_| {
                let service = service.clone();
                async move {
                    Ok::<_, hyper::Error>(hyper::service::service_fn(move |req: Request<Body>| {
                        let service = service.clone();
                        async move {
                            match req.uri().path() {
                                "/send" => {
                                    let body: hyper::body::Bytes =
                                        hyper::body::to_bytes(req.into_body()).await?;
                                    service.message_queue.sender.send_async(body).await.unwrap();
                                    Ok::<_, hyper::Error>(Response::new(hyper::body::Body::empty()))
                                }
                                "/recv" => {
                                    let body =
                                        service.message_queue.receiver.recv_async().await.unwrap();
                                    Ok::<_, hyper::Error>(Response::new(Body::from(body)))
                                }
                                _ => {
                                    tracing::error!("unsupported uri for persia message queue");
                                    let mut resp = Response::default();
                                    *resp.status_mut() = hyper::http::StatusCode::BAD_REQUEST;
                                    Ok(resp)
                                }
                            }
                        }
                    }))
                }
            }));

        let server_handler = Arc::new(persia_futures::tokio::task::spawn(
            async move { server.await },
        ));

        Self {
            server_handler,
            message_queue,
        }
    }

    pub async fn send(&self, content: Vec<u8>) {
        self.message_queue
            .sender
            .send_async(hyper::body::Bytes::from(content))
            .await
            .unwrap()
    }

    pub async fn recv(&self) -> Vec<u8> {
        self.message_queue
            .receiver
            .recv_async()
            .await
            .unwrap()
            .to_vec()
    }

    pub async fn handler(&self) -> Arc<persia_futures::tokio::task::JoinHandle<hyper::Result<()>>> {
        self.server_handler.clone()
    }
}
