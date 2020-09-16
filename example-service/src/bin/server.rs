use example_service::Service;
use hyper::service::{make_service_fn, };
use hyper::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let service =
        make_service_fn(|_| async { Ok::<_, hyper::Error>(Service {}) });

    let server = Server::bind(&([0, 0, 0, 0], 8080).into())
        .http2_only(true)
        .serve(service);

    server.await?;
    Ok(())
}
