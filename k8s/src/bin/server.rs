use persia_operator::crd::PersiaJobSpec;
use persia_operator::PersiaJobResources;

use actix_web::{get, post, web, App, HttpRequest, HttpServer, Responder};
use kube::client::Client;
use serde::{Deserialize, Serialize};
use structopt::StructOpt;

static KUBERNETES_CLIENT: once_cell::sync::OnceCell<Client> = once_cell::sync::OnceCell::new();

#[derive(Deserialize, Serialize)]
struct JobIdentifier {
    pub job_name: String,
    pub namespace: String,
}

#[derive(Deserialize, Serialize)]
struct PodIdentifier {
    pub pod_name: String,
    pub namespace: String,
}

#[derive(Deserialize, Serialize)]
struct ApplyRequest {
    pub job_identifier: JobIdentifier,
    pub spec: PersiaJobSpec,
}

#[derive(Deserialize, Serialize)]
struct ExecutionResults {
    pub success: bool,
    pub err_msg: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct ListResponse {
    pub execution_results: ExecutionResults,
    pub resources: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize)]
struct PodsResponse {
    pub execution_results: ExecutionResults,
    pub body: Option<String>,
}

#[post("/apply")]
async fn apply(req: web::Json<ApplyRequest>) -> impl Responder {
    let kubernetes_client = KUBERNETES_CLIENT
        .get()
        .expect("KUBERNETES_CLIENT not set")
        .clone();
    let resources = PersiaJobResources::new(
        &req.spec,
        &req.job_identifier.job_name,
        &req.job_identifier.namespace,
        kubernetes_client,
    );
    let resp = match resources.apply().await {
        Ok(_) => ExecutionResults {
            success: true,
            err_msg: None,
        },
        Err(e) => ExecutionResults {
            success: false,
            err_msg: Some(e.to_string()),
        },
    };
    serde_json::to_string(&resp)
}

#[post("/delete")]
async fn delete(req: web::Json<JobIdentifier>) -> impl Responder {
    let kubernetes_client = KUBERNETES_CLIENT
        .get()
        .expect("KUBERNETES_CLIENT not set")
        .clone();

    let resp = match PersiaJobResources::delete_resources(
        kubernetes_client,
        &req.namespace,
        &req.job_name,
    )
    .await
    {
        Ok(_) => ExecutionResults {
            success: true,
            err_msg: None,
        },
        Err(e) => ExecutionResults {
            success: false,
            err_msg: Some(e.to_string()),
        },
    };

    serde_json::to_string(&resp)
}

#[get("/listpods")]
async fn listpods(req: web::Json<JobIdentifier>) -> impl Responder {
    let kubernetes_client = KUBERNETES_CLIENT
        .get()
        .expect("KUBERNETES_CLIENT not set")
        .clone();

    let resp =
        match PersiaJobResources::get_pods_name(kubernetes_client, &req.namespace, &req.job_name)
            .await
        {
            Ok(pods) => ListResponse {
                execution_results: ExecutionResults {
                    success: true,
                    err_msg: None,
                },
                resources: Some(pods),
            },
            Err(e) => ListResponse {
                execution_results: ExecutionResults {
                    success: false,
                    err_msg: Some(e.to_string()),
                },
                resources: None,
            },
        };

    serde_json::to_string(&resp)
}

#[get("/listjobs")]
async fn listjobs(req: HttpRequest) -> impl Responder {
    let kubernetes_client = KUBERNETES_CLIENT
        .get()
        .expect("KUBERNETES_CLIENT not set")
        .clone();

    let namespace = req.match_info().get("namespace").unwrap_or("default");

    let resp = match PersiaJobResources::get_jobs_name(kubernetes_client, namespace).await {
        Ok(jobs) => ListResponse {
            execution_results: ExecutionResults {
                success: true,
                err_msg: None,
            },
            resources: Some(jobs),
        },
        Err(e) => ListResponse {
            execution_results: ExecutionResults {
                success: false,
                err_msg: Some(e.to_string()),
            },
            resources: None,
        },
    };

    serde_json::to_string(&resp)
}

#[get("/podstatus")]
async fn podstatus(req: web::Json<PodIdentifier>) -> impl Responder {
    let kubernetes_client = KUBERNETES_CLIENT
        .get()
        .expect("KUBERNETES_CLIENT not set")
        .clone();

    let resp =
        match PersiaJobResources::get_pod_status(kubernetes_client, &req.namespace, &req.pod_name)
            .await
        {
            Ok(s) => {
                let status = match s {
                    Some(pod_status) => serde_json::to_string(&pod_status).unwrap(),
                    None => String::from("None"),
                };
                PodsResponse {
                    execution_results: ExecutionResults {
                        success: true,
                        err_msg: None,
                    },
                    body: Some(status),
                }
            }
            Err(e) => PodsResponse {
                execution_results: ExecutionResults {
                    success: false,
                    err_msg: Some(e.to_string()),
                },
                body: None,
            },
        };

    serde_json::to_string(&resp)
}

#[derive(Debug, StructOpt, Clone)]
#[structopt()]
struct Cli {
    #[structopt(long)]
    port: u16,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    openssl_sys::init();
    let args: Cli = Cli::from_args();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("LOG_LEVEL"))
        .init();

    let kubernetes_client: Client = Client::try_default()
        .await
        .expect("Expected a valid KUBECONFIG environment variable.");

    if let Err(_) = KUBERNETES_CLIENT.set(kubernetes_client) {
        tracing::error!("set KUBERNETES_CLIENT muti times");
    }

    HttpServer::new(|| {
        App::new()
            .service(apply)
            .service(delete)
            .service(listpods)
            .service(podstatus)
            .service(listjobs)
    })
    .bind(("127.0.0.1", args.port))?
    .run()
    .await
}
