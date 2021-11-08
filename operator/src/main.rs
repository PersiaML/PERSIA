mod crd;
mod finalizer;
mod op;
mod utils;

use crate::crd::PersiaJob;
use futures::stream::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::CustomResourceExt;
use kube::Resource;
use kube::ResourceExt;
use kube::{api::ListParams, client::Client, Api};
use kube_runtime::controller::{Context, ReconcilerAction};
use kube_runtime::Controller;
use parking_lot::Mutex;
use std::collections::HashMap;
use tokio::time::Duration;

#[tokio::main]
async fn main() {
    if std::env::var("GEN_CRD")
        .unwrap_or(String::from("false"))
        .parse::<bool>()
        .expect("GEN_CRD should be true or false")
    {
        print!("{}", serde_yaml::to_string(&PersiaJob::crd()).unwrap());
    }

    let kubernetes_client: Client = Client::try_default()
        .await
        .expect("Expected a valid KUBECONFIG environment variable.");

    let crd_api: Api<PersiaJob> = Api::all(kubernetes_client.clone());
    let context: Context<ContextData> = Context::new(ContextData::new(kubernetes_client.clone()));

    Controller::new(crd_api.clone(), ListParams::default())
        .run(reconcile, on_error, context)
        .for_each(|reconciliation_result| async move {
            match reconciliation_result {
                Ok(persia_resource) => {
                    println!("Reconciliation successful. Resource: {:?}", persia_resource);
                }
                Err(reconciliation_err) => {
                    eprintln!("Reconciliation error: {:?}", reconciliation_err)
                }
            }
        })
        .await;
}

struct ContextData {
    client: Client,
    jobs: Mutex<HashMap<String, Vec<Pod>>>,
}

impl ContextData {
    pub fn new(client: Client) -> Self {
        ContextData {
            client,
            jobs: Mutex::new(HashMap::new()),
        }
    }
}

enum Action {
    Create,
    Delete,
    NoOp,
}

async fn reconcile(
    job: PersiaJob,
    context: Context<ContextData>,
) -> Result<ReconcilerAction, Error> {
    let client: Client = context.get_ref().client.clone();

    let namespace: String = match job.namespace() {
        None => {
            return Err(Error::UserInputError(
                "Expected PersiaJob resource to be namespaced. Can't deploy to an unknown namespace."
                    .to_owned(),
            ));
        }
        Some(namespace) => namespace,
    };

    return match determine_action(&job) {
        Action::Create => {
            let name = job.name();
            eprintln!("Creating PersiaJob: {}", name);

            finalizer::add(client.clone(), &name, &namespace).await?;

            let pods: Vec<Pod> = job.spec.gen_pods(&name, &namespace);
            op::deploy(client, &pods, &namespace).await?;

            let mut jobs = context.get_ref().jobs.lock();
            jobs.insert(name, pods);

            Ok(ReconcilerAction {
                requeue_after: Some(Duration::from_secs(10)),
            })
        }
        Action::Delete => {
            let name = job.name();
            eprintln!("Deletding PersiaJob: {}", name);

            let pods_name: Vec<String> = job.spec.gen_pods_name(name.as_str());

            op::delete(client.clone(), &pods_name, &namespace).await?;

            finalizer::delete(client, &name, &namespace).await?;

            let mut jobs = context.get_ref().jobs.lock();
            jobs.remove(&name);

            Ok(ReconcilerAction {
                requeue_after: None,
            })
        }
        Action::NoOp => Ok(ReconcilerAction {
            requeue_after: Some(Duration::from_secs(10)),
        }),
    };
}

fn determine_action(job: &PersiaJob) -> Action {
    return if job.meta().deletion_timestamp.is_some() {
        Action::Delete
    } else if job
        .meta()
        .finalizers
        .as_ref()
        .map_or(true, |finalizers| finalizers.is_empty())
    {
        Action::Create
    } else {
        Action::NoOp
    };
}

fn on_error(error: &Error, _context: Context<ContextData>) -> ReconcilerAction {
    eprintln!("Reconciliation error:\n{:?}", error);
    ReconcilerAction {
        requeue_after: Some(Duration::from_secs(5)),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Kubernetes reported error: {source}")]
    KubeError {
        #[from]
        source: kube::Error,
    },
    #[error("Invalid PersiaJob CRD: {0}")]
    UserInputError(String),
}
