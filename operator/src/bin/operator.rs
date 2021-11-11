use futures::stream::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use k8s_openapi::api::core::v1::Service;
use kube::Resource;
use kube::ResourceExt;
use kube::{api::ListParams, client::Client, Api};
use kube_runtime::controller::{Context, ReconcilerAction};
use kube_runtime::Controller;
use persia_operator::crd::PersiaJob;
use persia_operator::{finalizer, utils};
use tokio::time::Duration;

#[tokio::main]
async fn main() {
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
}

impl ContextData {
    pub fn new(client: Client) -> Self {
        ContextData { client }
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
            utils::deploy_pods(client.clone(), &pods, &namespace).await?;

            let services: Vec<Service> = job.spec.gen_services(&name, &namespace);
            utils::deploy_services(client, &services, &namespace).await?;

            Ok(ReconcilerAction {
                requeue_after: Some(Duration::from_secs(10)),
            })
        }
        Action::Delete => {
            let name = job.name();
            eprintln!("Deletding PersiaJob: {}", name);

            let services: Vec<Service> = job.spec.gen_services(&name, &namespace);
            let mut services_name = Vec::new();
            services.into_iter().for_each(|s| {
                if let Some(service_name) = s.metadata.name {
                    services_name.push(service_name);
                }
            });

            utils::delete_services(client.clone(), &services_name, &namespace).await?;

            let pods: Vec<Pod> = job.spec.gen_pods(&name, &namespace);
            let mut pods_name = Vec::new();
            pods.into_iter().for_each(|p| {
                if let Some(pod_name) = p.metadata.name {
                    pods_name.push(pod_name);
                }
            });

            utils::delete_pods(client.clone(), &pods_name, &namespace).await?;

            finalizer::delete(client, &name, &namespace).await?;

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
