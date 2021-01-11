use example_service::Service;
use hyper::service::make_service_fn;
use hyper::Server;
use example_service::{Input, Output};
use structopt::StructOpt;
use std::time::Duration;
use std::sync::Arc;

#[derive(Debug, StructOpt, Clone)]
#[structopt()]
struct Cli {
    #[structopt(long)]
    port: u16,
    #[structopt(long)]
    server_addrs: Vec<String>,
    #[structopt(long)]
    num_tasks: usize,
}

// TODO: impl Readable Writable for RecyclableVec<VecPool> ?

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("LOG_LEVEL"))
        .init();

    for i in 0..10 {
        let mut req = vec![0u8; 573741824];
        let mut req2 = vec![0u8; 573741824];
        let start_time = std::time::Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(req.as_ptr(), req2.as_mut_ptr(), req.len());
        }
        println!("memcpy speed {}M per second", req.len() as f32/ 1024. / 1024. / start_time.elapsed().as_millis() as f32 * 1000.);
        req[10] = i as u8;
    }

    let mut req = vec![0u8; 573741824];
    let mut req2 = vec![0u8; 573741824];
    for i in 0..10 {
        let start_time = std::time::Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(req.as_ptr(), req2.as_mut_ptr(), req.len());
        }
        println!("memcpy no page fault speed {}M per second", req.len() as f32/ 1024. / 1024. / start_time.elapsed().as_millis() as f32 * 1000.);
        req[10] = i as u8;
    }

    let args: Cli = Cli::from_args();

    let service =
        make_service_fn(|_| async { Ok::<_, hyper::Error>(Service {}) });

    let server = Server::bind(&([0, 0, 0, 0], args.port).into())
        // .http2_only(true)
        // .tcp_nodelay(true)
        .http1_only(true)
        .serve(service);

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("rt build");
        rt.block_on(server).unwrap();
    });

    for server_addr in &args.server_addrs {
        let client = persia_rpc::RpcClient::new(format!("{}:{}", server_addr, args.port).as_str())?;
        let rpc_client = Arc::new(example_service::ServiceClient::new(client));
        for _ in 0..args.num_tasks {
            let rpc_client = rpc_client.clone();
            tokio::spawn(
                async move {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    let req = vec![0.0f32; 20971520];
                    let mut start_time = std::time::Instant::now();
                    for i in 0..10000usize {
                        let _result = rpc_client.large_body_rpc_test(&req).await.unwrap();
                        if i % 10 == 0 {
                            println!("{}ms per 10 reqests", start_time.elapsed().as_millis());
                            start_time = std::time::Instant::now();
                        }
                    }
                }
            );
        }
    }

    std::thread::sleep(Duration::from_secs(3600));

    // let client = persia_rpc::RpcClient::new("127.0.0.1:8080")?;
    // let rpc_client = example_service::ServiceClient::new(client);


    // for _ in 0..10 {
    //     let _result: Output = dbg!(
    //         rpc_client
    //             .rpc_test(Input { msg: "haha".into() })
    //             .await
    //     ).unwrap();
    // }
    //
    // for _ in 0..10 {
    //     let _result: Output = dbg!(
    //         rpc_client
    //             .rpc_test_compressed(Input { msg: "haha".into() })
    //             .await
    //     ).unwrap();
    // }
    //
    // for _ in 0..10 {
    //     let _result: Output = dbg!(
    //         rpc_client
    //             .rpc_test_2()
    //             .await
    //     ).unwrap();
    // }
    //

    Ok(())
}
