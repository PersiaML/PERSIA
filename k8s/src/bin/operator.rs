use futures::stream::StreamExt;
use kube::Resource;
use kube::ResourceExt;
use kube::{api::ListParams, client::Client, Api};
use kube_runtime::controller::{Context, ReconcilerAction};
use kube_runtime::Controller;
use tokio::time::Duration;

use persia_operator::crd::PersiaJob;
use persia_operator::error::Error;
use persia_operator::finalizer;
use persia_operator::PersiaJobResources;

#[tokio::main]
async fn main() {
    openssl_sys::init();

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

    let job_resources = PersiaJobResources::new(&job.spec, &job.name(), &namespace, client.clone());

    return match determine_action(&job) {
        Action::Create => {
            let job_name = job.name();
            eprintln!("Creating PersiaJob: {}", job_name);

            finalizer::add(client.clone(), &job_name, &namespace).await?;

            job_resources.apply().await?;

            Ok(ReconcilerAction {
                requeue_after: Some(Duration::from_secs(10)),
            })
        }
        Action::Delete => {
            let job_name = job.name();
            eprintln!("Deleting PersiaJob: {}", job_name);

            job_resources.delete().await?;
            finalizer::delete(client, &job_name, &namespace).await?;

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
