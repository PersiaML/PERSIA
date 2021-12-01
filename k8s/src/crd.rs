use k8s_openapi::api::core::v1::{
    Container, EnvVar, Pod, PodSpec, ResourceRequirements, Service, ServicePort, ServiceSpec,
    Volume, VolumeMount,
};
use kube::api::ObjectMeta;
use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DEFAULT_CUDA_IMAGE: &str = "persiaml/persia-cuda-runtime:latest";
pub const DEFAULT_CPU_IMAGE: &str = "persiaml/persia-cpu-runtime:latest";

pub fn get_embedding_ps_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-embedding-ps-{}", job_name, replica_index)
}

pub fn get_embedding_worker_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-embedding-worker-{}", job_name, replica_index)
}

pub fn get_nn_worker_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-nn-worker-{}", job_name, replica_index)
}

pub fn get_dataloader_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-dataloader-{}", job_name, replica_index)
}

pub fn get_metrics_gateway_pod_name(job_name: &str) -> String {
    format!("{}-metrics-gateway", job_name)
}

pub fn get_metrics_gateway_service_name(job_name: &str) -> String {
    format!("{}-metrics-gateway-service", job_name)
}

pub fn get_label_selector(job_name: &str) -> String {
    format!("persia_job={}", job_name)
}

#[derive(CustomResource, Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
#[kube(
    group = "persia.com",
    version = "v1",
    kind = "PersiaJob",
    plural = "persiajobs",
    derive = "PartialEq",
    namespaced
)]
#[allow(non_snake_case)]
pub struct PersiaJobSpec {
    pub persiaEnv: PersiaEnvSpec,
    pub enableMetrics: Option<bool>,
    pub volumes: Option<Vec<Volume>>,
    pub env: Option<Vec<EnvVar>>,
    pub logLevel: Option<String>,
    pub embeddingParameterServer: Option<EmbeddingParameterServerSpec>,
    pub embeddingWorker: Option<EmbeddingWorkerSpec>,
    pub nnWorker: Option<NNWorkerSpec>,
    pub dataloader: Option<DataLoaderSpec>,
    pub restartPolicy: Option<String>,
    pub imagePullPolicy: Option<String>,
}

impl PersiaJobSpec {
    fn gen_pod_template(&self, job_name: &str, namespace: &str) -> Pod {
        let log_level = self.logLevel.clone().unwrap_or(String::from("info"));
        let restart_policy = self.restartPolicy.clone().unwrap_or(String::from("Never"));
        let image_pull_policy = self
            .imagePullPolicy
            .clone()
            .unwrap_or(String::from("Always"));

        let mut labels: BTreeMap<String, String> = BTreeMap::new();
        labels.insert("persia_job".to_owned(), job_name.to_owned());

        let mut env = vec![
            EnvVar {
                name: String::from("PERSIA_EMBEDDING_CONFIG"),
                value: Some(self.persiaEnv.PERSIA_EMBEDDING_CONFIG.clone()),
                ..EnvVar::default()
            },
            EnvVar {
                name: String::from("PERSIA_GLOBAL_CONFIG"),
                value: Some(self.persiaEnv.PERSIA_GLOBAL_CONFIG.clone()),
                ..EnvVar::default()
            },
            EnvVar {
                name: String::from("LOG_LEVEL"),
                value: Some(log_level),
                ..EnvVar::default()
            },
            EnvVar {
                name: String::from("RUST_BACKTRACE"),
                value: Some(String::from("full")),
                ..EnvVar::default()
            },
            EnvVar {
                name: String::from("PERSIA_METRICS_GATEWAY_ADDR"),
                value: Some(format!(
                    "{}:9091",
                    get_metrics_gateway_service_name(job_name)
                )),
                ..EnvVar::default()
            },
        ];

        if let Some(e) = &self.env {
            e.iter().for_each(|env_var| {
                env.push(env_var.clone());
            });
        }

        let pod_spec = PodSpec {
            containers: vec![Container {
                command: Some(vec!["persia-launcher".to_string()]),
                env: Some(env),
                image_pull_policy: Some(image_pull_policy),
                ..Container::default()
            }],
            volumes: self.volumes.clone(),
            restart_policy: Some(restart_policy),
            ..PodSpec::default()
        };

        Pod {
            metadata: ObjectMeta {
                namespace: Some(namespace.to_owned()),
                labels: Some(labels.clone()),
                ..ObjectMeta::default()
            },
            spec: Some(pod_spec),
            ..Pod::default()
        }
    }

    pub fn gen_services(&self, job_name: &str, namespace: &str) -> Vec<Service> {
        let mut results = Vec::new();

        let mut labels: BTreeMap<String, String> = BTreeMap::new();
        labels.insert("persia_job".to_owned(), job_name.to_owned());

        if self.enableMetrics.unwrap_or(true) {
            let service_name = get_metrics_gateway_service_name(job_name);

            let mut selector: BTreeMap<String, String> = BTreeMap::new();
            selector.insert("service".to_owned(), service_name.clone());

            let metrics_gateway_service = Service {
                metadata: ObjectMeta {
                    name: Some(service_name.clone()),
                    namespace: Some(namespace.to_owned()),
                    labels: Some(labels.clone()),
                    ..ObjectMeta::default()
                },
                spec: Some(ServiceSpec {
                    ports: Some(vec![ServicePort {
                        port: 9091,
                        ..ServicePort::default()
                    }]),
                    selector: Some(selector),
                    ..ServiceSpec::default()
                }),
                ..Service::default()
            };

            results.push(metrics_gateway_service);
        }

        results
    }

    pub fn gen_pods(&self, job_name: &str, namespace: &str) -> Vec<Pod> {
        let mut results = Vec::new();

        if let Some(embedding_server) = &self.embeddingParameterServer {
            let mut emb_server_spec: Vec<Pod> = (0..embedding_server.replicas)
                .into_iter()
                .map(|replica_idx| {
                    let mut pod = self.gen_pod_template(job_name, namespace);

                    let podspec = pod.spec.as_mut().unwrap();
                    let container = podspec.containers.first_mut().unwrap();

                    container.name = "emb-server".to_string();
                    container.args = Some(
                        vec![
                            "embedding-parameter-server",
                            "--embedding-config",
                            "$(PERSIA_EMBEDDING_CONFIG)",
                            "--global-config",
                            "$(PERSIA_GLOBAL_CONFIG)",
                            "--replica-index",
                            "$(REPLICA_INDEX)",
                            "--replica-size",
                            "$(REPLICA_SIZE)",
                        ]
                        .into_iter()
                        .map(|x| x.to_string())
                        .collect(),
                    );

                    container.resources = embedding_server.resources.clone();
                    container.volume_mounts = embedding_server.volumeMounts.clone();

                    container.image = embedding_server.image.clone();
                    if container.image.is_none() {
                        container.image = Some(String::from(DEFAULT_CPU_IMAGE));
                    }

                    let env = container.env.as_mut().unwrap();
                    env.push(EnvVar {
                        name: String::from("REPLICA_INDEX"),
                        value: Some(replica_idx.to_string()),
                        ..EnvVar::default()
                    });

                    env.push(EnvVar {
                        name: String::from("REPLICA_SIZE"),
                        value: Some(embedding_server.replicas.to_string()),
                        ..EnvVar::default()
                    });

                    if let Some(e) = &embedding_server.env {
                        e.iter().for_each(|env_var| {
                            env.push(env_var.clone());
                        });
                    }

                    pod.metadata.name = Some(get_embedding_ps_pod_name(job_name, replica_idx));
                    pod
                })
                .collect();

            results.append(&mut emb_server_spec);
        }

        if let Some(embedding_worker) = &self.embeddingWorker {
            let mut embedding_worker_spec: Vec<Pod> = (0..embedding_worker.replicas)
                .into_iter()
                .map(|replica_idx| {
                    let mut pod = self.gen_pod_template(job_name, namespace);
                    let podspec = pod.spec.as_mut().unwrap();
                    let container = podspec.containers.first_mut().unwrap();

                    container.name = "embedding-worker".to_string();
                    container.args = Some(
                        vec![
                            "embedding-worker",
                            "--embedding-config",
                            "$(PERSIA_EMBEDDING_CONFIG)",
                            "--global-config",
                            "$(PERSIA_GLOBAL_CONFIG)",
                            "--replica-index",
                            "$(REPLICA_INDEX)",
                            "--replica-size",
                            "$(REPLICA_SIZE)",
                        ]
                        .into_iter()
                        .map(|x| x.to_string())
                        .collect(),
                    );

                    container.resources = embedding_worker.resources.clone();
                    container.volume_mounts = embedding_worker.volumeMounts.clone();

                    container.image = embedding_worker.image.clone();
                    if container.image.is_none() {
                        container.image = Some(String::from(DEFAULT_CPU_IMAGE));
                    }

                    let env = container.env.as_mut().unwrap();
                    env.push(EnvVar {
                        name: String::from("REPLICA_INDEX"),
                        value: Some(replica_idx.to_string()),
                        ..EnvVar::default()
                    });

                    env.push(EnvVar {
                        name: String::from("REPLICA_SIZE"),
                        value: Some(embedding_worker.replicas.to_string()),
                        ..EnvVar::default()
                    });

                    if let Some(e) = &embedding_worker.env {
                        e.iter().for_each(|env_var| {
                            env.push(env_var.clone());
                        });
                    }

                    pod.metadata.name = Some(get_embedding_worker_pod_name(job_name, replica_idx));
                    pod
                })
                .collect();

            results.append(&mut embedding_worker_spec);
        }

        if let Some(nn_worker) = &self.nnWorker {
            let mut nn_worker_spec: Vec<Pod> = (0..nn_worker.replicas)
                .into_iter()
                .map(|replica_idx| {
                    let mut pod = self.gen_pod_template(job_name, namespace);
                    let podspec = pod.spec.as_mut().unwrap();

                    let container = podspec.containers.first_mut().unwrap();

                    container.name = "nn-worker".to_string();
                    container.args = Some(
                        vec![
                            "nn-worker",
                            "$(PERSIA_NN_WORKER_ENTRY)",
                            "--nproc-per-node",
                            "$(NPROC_PER_NODE)",
                            "--nnodes",
                            "$(REPLICA_SIZE)",
                            "--node-rank",
                            "$(REPLICA_INDEX)",
                        ]
                        .into_iter()
                        .map(|x| x.to_string())
                        .collect(),
                    );

                    container.resources = nn_worker.resources.clone();
                    container.volume_mounts = nn_worker.volumeMounts.clone();

                    container.image = nn_worker.image.clone();
                    if container.image.is_none() {
                        container.image = Some(String::from(DEFAULT_CUDA_IMAGE));
                    }

                    let env = container.env.as_mut().unwrap();
                    env.push(EnvVar {
                        name: String::from("REPLICA_INDEX"),
                        value: Some(replica_idx.to_string()),
                        ..EnvVar::default()
                    });
                    env.push(EnvVar {
                        name: String::from("REPLICA_SIZE"),
                        value: Some(nn_worker.replicas.to_string()),
                        ..EnvVar::default()
                    });
                    env.push(EnvVar {
                        name: String::from("NPROC_PER_NODE"),
                        value: Some(nn_worker.nprocPerNode.to_string()),
                        ..EnvVar::default()
                    });
                    env.push(EnvVar {
                        name: String::from("PERSIA_NN_WORKER_ENTRY"),
                        value: self.persiaEnv.PERSIA_NN_WORKER_ENTRY.clone(),
                        ..EnvVar::default()
                    });

                    if let Some(e) = &nn_worker.env {
                        e.iter().for_each(|env_var| {
                            env.push(env_var.clone());
                        });
                    }

                    pod.metadata.name = Some(get_nn_worker_pod_name(job_name, replica_idx));
                    pod
                })
                .collect();

            results.append(&mut nn_worker_spec);
        }

        if let Some(dataloader) = &self.dataloader {
            let mut dataloader_spec: Vec<Pod> = (0..dataloader.replicas)
                .into_iter()
                .map(|replica_idx| {
                    let mut pod = self.gen_pod_template(job_name, namespace);
                    let podspec = pod.spec.as_mut().unwrap();
                    let container = &mut podspec
                        .containers
                        .first_mut()
                        .expect("no containers in a persia podspec template");

                    container.name = "data-loader".to_string();
                    container.args = Some(
                        vec![
                            "data-loader",
                            "$(PERSIA_DATALOADER_ENTRY)",
                            "--replica-index",
                            "$(REPLICA_INDEX)",
                            "--replica-size",
                            "$(REPLICA_SIZE)",
                        ]
                        .into_iter()
                        .map(|x| x.to_string())
                        .collect(),
                    );

                    container.resources = dataloader.resources.clone();
                    container.volume_mounts = dataloader.volumeMounts.clone();

                    container.image = dataloader.image.clone();
                    if container.image.is_none() {
                        container.image = Some(String::from(DEFAULT_CPU_IMAGE));
                    }

                    let env = container.env.as_mut().unwrap();
                    env.push(EnvVar {
                        name: String::from("REPLICA_INDEX"),
                        value: Some(replica_idx.to_string()),
                        ..EnvVar::default()
                    });
                    env.push(EnvVar {
                        name: String::from("REPLICA_SIZE"),
                        value: Some(dataloader.replicas.to_string()),
                        ..EnvVar::default()
                    });
                    env.push(EnvVar {
                        name: String::from("PERSIA_DATALOADER_ENTRY"),
                        value: self.persiaEnv.PERSIA_DATALOADER_ENTRY.clone(),
                        ..EnvVar::default()
                    });

                    if let Some(e) = &dataloader.env {
                        e.iter().for_each(|env_var| {
                            env.push(env_var.clone());
                        });
                    }

                    pod.metadata.name = Some(get_dataloader_pod_name(job_name, replica_idx));
                    pod
                })
                .collect();

            results.append(&mut dataloader_spec);
        }

        if self.enableMetrics.unwrap_or(true) {
            let service_name = get_metrics_gateway_service_name(job_name);

            let mut metrics_labels: BTreeMap<String, String> = BTreeMap::new();
            metrics_labels.insert("persia_job".to_owned(), job_name.to_owned());
            metrics_labels.insert("service".to_owned(), service_name);
            metrics_labels.insert("prom".to_owned(), job_name.to_owned());

            let metrics_pod = Pod {
                metadata: ObjectMeta {
                    name: Some(get_metrics_gateway_pod_name(job_name)),
                    namespace: Some(namespace.to_owned()),
                    labels: Some(metrics_labels),
                    ..ObjectMeta::default()
                },
                spec: Some(PodSpec {
                    containers: vec![Container {
                        name: String::from("pushgateway"),
                        image: Some(String::from("prom/pushgateway:latest")),
                        image_pull_policy: Some(String::from("IfNotPresent")),
                        ..Container::default()
                    }],
                    volumes: self.volumes.clone(),
                    ..PodSpec::default()
                }),
                ..Pod::default()
            };

            results.push(metrics_pod);
        }

        results
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
#[allow(non_snake_case)]
pub struct EmbeddingParameterServerSpec {
    pub replicas: usize,
    pub resources: Option<ResourceRequirements>,
    pub volumeMounts: Option<Vec<VolumeMount>>,
    pub env: Option<Vec<EnvVar>>,
    pub image: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
#[allow(non_snake_case)]
pub struct EmbeddingWorkerSpec {
    pub replicas: usize,
    pub resources: Option<ResourceRequirements>,
    pub volumeMounts: Option<Vec<VolumeMount>>,
    pub env: Option<Vec<EnvVar>>,
    pub image: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
#[allow(non_snake_case)]
pub struct NNWorkerSpec {
    pub replicas: usize,
    pub nprocPerNode: usize,
    pub resources: Option<ResourceRequirements>,
    pub volumeMounts: Option<Vec<VolumeMount>>,
    pub env: Option<Vec<EnvVar>>,
    pub image: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
#[allow(non_snake_case)]
pub struct DataLoaderSpec {
    pub replicas: usize,
    pub resources: Option<ResourceRequirements>,
    pub volumeMounts: Option<Vec<VolumeMount>>,
    pub env: Option<Vec<EnvVar>>,
    pub image: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
#[allow(non_snake_case)]
pub struct PersiaEnvSpec {
    pub PERSIA_GLOBAL_CONFIG: String,
    pub PERSIA_EMBEDDING_CONFIG: String,
    pub PERSIA_NN_WORKER_ENTRY: Option<String>,
    pub PERSIA_DATALOADER_ENTRY: Option<String>,
}
