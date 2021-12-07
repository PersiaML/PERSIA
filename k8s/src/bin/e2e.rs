use std::time::Duration;

use collection_macros::btreemap;
use k8s_openapi::api::core::v1::{EnvVar, ResourceRequirements};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use kube::client::Client;
use persia_operator::crd::{
    get_nn_worker_pod_name, DataLoaderSpec, EmbeddingParameterServerSpec, EmbeddingWorkerSpec,
    NNWorkerSpec, PersiaEnvSpec, PersiaJobSpec,
};
use persia_operator::PersiaJobResources;

const EMBEDDING_PS_REPLICAS: usize = 2;
const EMBEDDING_WORKER_REPLICAS: usize = 2;
const NN_WORKER_REPLICAS: usize = 2;
const NPROC_PER_NODE: usize = 2;
const DATALOADER_REPLICAS: usize = 1;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let kubernetes_client: Client = Client::try_default()
        .await
        .expect("Expected a valid KUBECONFIG environment variable.");

    let buildkite_pipeline_id =
        std::env::var("BUILDKITE_PIPELINE_ID").expect("Failed to get BUILDKITE_PIPELINE_ID");

    let job_name = format!("persia-ci-{}", buildkite_pipeline_id);
    let namespace = "default";
    let job_spec = gen_spec();
    let persia_job =
        PersiaJobResources::new(&job_spec, &job_name, &namespace, kubernetes_client.clone());

    persia_job.delete().await;
    
    persia_job.apply().await?;

    let result = wait_for_compelete(
        &job_name,
        &namespace,
        kubernetes_client.clone(),
        Duration::from_secs(600),
    )
    .await;

    persia_job.delete().await?;
    result
}

fn gen_spec() -> PersiaJobSpec {
    let buildkite_pipeline_id =
        std::env::var("BUILDKITE_PIPELINE_ID").expect("Failed to get BUILDKITE_PIPELINE_ID");

    let cpu_image = format!("persia-cpu-runtime:{}", buildkite_pipeline_id);
    let cuda_image = format!("persia-cuda-runtime:{}", buildkite_pipeline_id);

    PersiaJobSpec {
        persiaEnv: PersiaEnvSpec {
            PERSIA_GLOBAL_CONFIG: String::from(
                "/home/PERSIA/examples/src/adult-income/config/global_config.yml",
            ),
            PERSIA_EMBEDDING_CONFIG: String::from(
                "/home/PERSIA/examples/src/adult-income/config/embedding_config.yml",
            ),
            PERSIA_NN_WORKER_ENTRY: Some(String::from(
                "/home/PERSIA/examples/src/adult-income/train.py",
            )),
            PERSIA_DATALOADER_ENTRY: Some(String::from(
                "/home/PERSIA/examples/src/adult-income/data_loader.py",
            )),
        },
        enableMetrics: Some(false),
        volumes: None,
        env: Some(vec![EnvVar {
            name: String::from("PERSIA_NATS_URL"),
            value: Some(format!(
                "nats://persia-nats-service-{}:4222",
                buildkite_pipeline_id
            )),
            ..EnvVar::default()
        }]),
        logLevel: None,
        embeddingParameterServer: Some(EmbeddingParameterServerSpec {
            replicas: EMBEDDING_PS_REPLICAS,
            resources: Some(ResourceRequirements {
                limits: Some(btreemap! {
                    String::from("cpu") => Quantity(String::from("2")),
                    String::from("memory") => Quantity(String::from("12Gi")),
                }),
                ..ResourceRequirements::default()
            }),
            volumeMounts: None,
            env: None,
            image: Some(cpu_image.clone()),
        }),
        embeddingWorker: Some(EmbeddingWorkerSpec {
            replicas: EMBEDDING_WORKER_REPLICAS,
            resources: Some(ResourceRequirements {
                limits: Some(btreemap! {
                    String::from("cpu") => Quantity(String::from("2")),
                    String::from("memory") => Quantity(String::from("12Gi")),
                }),
                ..ResourceRequirements::default()
            }),
            volumeMounts: None,
            env: None,
            image: Some(cpu_image.clone()),
        }),
        nnWorker: Some(NNWorkerSpec {
            replicas: NN_WORKER_REPLICAS,
            nprocPerNode: NPROC_PER_NODE,
            resources: Some(ResourceRequirements {
                limits: Some(btreemap! {
                    String::from("cpu") => Quantity(String::from("2")),
                    String::from("memory") => Quantity(String::from("12Gi")),
                    String::from("nvidia.com/gpu") => Quantity(NPROC_PER_NODE.to_string()),
                }),
                ..ResourceRequirements::default()
            }),
            volumeMounts: None,
            env: Some(vec![
                EnvVar {
                    name: String::from("CUBLAS_WORKSPACE_CONFIG"),
                    value: Some(String::from(":4096:8")),
                    ..EnvVar::default()
                },
                EnvVar {
                    name: String::from("ENABLE_CUDA"),
                    value: Some(String::from("1")),
                    ..EnvVar::default()
                },
            ]),
            image: Some(cuda_image.clone()),
        }),
        dataloader: Some(DataLoaderSpec {
            replicas: DATALOADER_REPLICAS,
            resources: Some(ResourceRequirements {
                limits: Some(btreemap! {
                    String::from("cpu") => Quantity(String::from("2")),
                    String::from("memory") => Quantity(String::from("12Gi")),
                }),
                ..ResourceRequirements::default()
            }),
            volumeMounts: None,
            env: None,
            image: Some(cpu_image.clone()),
        }),
        restartPolicy: Some(String::from("Never")),
        imagePullPolicy: Some(String::from("Never")),
    }
}

async fn wait_for_compelete(
    job_name: &str,
    namespace: &str,
    kubernetes_client: Client,
    timeout: Duration,
) -> anyhow::Result<()> {
    let pods_name =
        PersiaJobResources::get_pods_name(kubernetes_client.clone(), namespace, job_name).await?;
    println!("Created Pods: {:?}", pods_name);

    let master_name = get_nn_worker_pod_name(job_name, 0);

    let start_time = std::time::Instant::now();

    loop {
        for pod_name in pods_name.iter() {
            let status =
                PersiaJobResources::get_pod_status(kubernetes_client.clone(), namespace, pod_name)
                    .await?
                    .ok_or(anyhow::anyhow!("Pod status is None"))?;
            let phase = status.phase.ok_or(anyhow::anyhow!("Pod phase is None"))?;
            println!("Status of Pod {} is {:?}", pod_name, phase);

            if "Failed" == phase.as_str() {
                let log =
                    PersiaJobResources::get_pod_log(kubernetes_client.clone(), namespace, pod_name)
                        .await?;
                eprintln!("Pod {} Failed, log:\n{}: ", pod_name, log);
                return Err(anyhow::anyhow!("Pod Failed"));
            }
        }

        let master_status =
            PersiaJobResources::get_pod_status(kubernetes_client.clone(), &namespace, &master_name)
                .await?
                .ok_or(anyhow::anyhow!("Pod status is None"))?;

        let master_phase = master_status
            .phase
            .ok_or(anyhow::anyhow!("Pod phase is None"))?;

        println!("Status of Pod {} is {:?}", master_name, master_phase);

        match master_phase.as_str() {
            "Succeeded" => {
                let log = PersiaJobResources::get_pod_log(
                    kubernetes_client.clone(),
                    namespace,
                    &master_name,
                )
                .await?;
                println!("Master Pod log:\n{}", log);
                println!("Master Pod Succeeded, start to clean up...");
                return Ok(());
            }
            _ => {}
        }

        if start_time.elapsed() > timeout {
            eprintln!("timeout for running system test");
            return Err(anyhow::anyhow!("Timeout"));
        }

        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
