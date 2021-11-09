use k8s_openapi::api::core::v1::{
    Container, EnvVar, Pod, PodSpec, ResourceRequirements, Volume, VolumeMount,
};
use kube::api::ObjectMeta;
use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::utils::{
    get_dataloader_pod_name, get_emb_server_pod_name, get_mid_server_pod_name, get_trainer_pod_name,
};

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
    pub persiaVersion: Option<String>,
    pub globalConfigPath: String,
    pub embeddingConfigPath: String,
    pub trainerPyEntryPath: String,
    pub dataLoaderPyEntryPath: Option<String>,
    pub volumes: Option<Vec<Volume>>,
    pub env: Option<Vec<EnvVar>>,
    pub logLevel: Option<String>,
    pub embeddingServer: Option<EmbeddingSpec>,
    pub middlewareServer: Option<MiddlewareSpec>,
    pub trainer: Option<TrainerSpec>,
    pub dataloader: Option<DataLoaderSpec>,
}

impl PersiaJobSpec {
    fn gen_podspec_template(&self) -> PodSpec {
        let persia_version = self.persiaVersion.clone().unwrap_or(String::from("latest"));
        let log_level = self.logLevel.clone().unwrap_or(String::from("info"));

        let mut env = vec![
            EnvVar {
                name: String::from("EMBEDDING_CONFIG_PATH"),
                value: Some(self.embeddingConfigPath.clone()),
                ..EnvVar::default()
            },
            EnvVar {
                name: String::from("GLOBAL_CONFIG_PATH"),
                value: Some(self.globalConfigPath.clone()),
                ..EnvVar::default()
            },
            EnvVar {
                name: String::from("TRAINER_PY_ENTRY_PATH"),
                value: Some(self.trainerPyEntryPath.clone()),
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
        ];

        if let Some(e) = &self.env {
            e.iter().for_each(|env_var| {
                env.push(env_var.clone());
            });
        }

        PodSpec {
            containers: vec![Container {
                command: Some(vec!["persia_launcher".to_string()]),
                image: Some(format!("persiaml/persia-cuda-runtime:{}", persia_version)),
                env: Some(env),
                ..Container::default()
            }],
            volumes: self.volumes.clone(),
            ..PodSpec::default()
        }
    }

    pub fn gen_pods_name(&self, job_name: &str) -> Vec<String> {
        let mut results = Vec::new();

        if let Some(embedding_server) = &self.embeddingServer {
            let mut emb_server_pod_name: Vec<String> = (0..embedding_server.replicas)
                .into_iter()
                .map(|replica_idx| get_emb_server_pod_name(job_name, replica_idx))
                .collect();

            results.append(&mut emb_server_pod_name);
        }

        if let Some(middleware_server) = &self.middlewareServer {
            let mut middleware_server_pod_name: Vec<String> = (0..middleware_server.replicas)
                .into_iter()
                .map(|replica_idx| get_mid_server_pod_name(job_name, replica_idx))
                .collect();

            results.append(&mut middleware_server_pod_name);
        }

        if let Some(trainer) = &self.trainer {
            let mut trainer_pod_name: Vec<String> = (0..trainer.replicas)
                .into_iter()
                .map(|replica_idx| get_trainer_pod_name(job_name, replica_idx))
                .collect();

            results.append(&mut trainer_pod_name);
        }

        if let Some(dataloader) = &self.dataloader {
            let mut dataloader_pod_name: Vec<String> = (0..dataloader.replicas)
                .into_iter()
                .map(|replica_idx| get_dataloader_pod_name(job_name, replica_idx))
                .collect();

            results.append(&mut dataloader_pod_name);
        }

        results
    }

    pub fn gen_pods(&self, job_name: &str, namespace: &str) -> Vec<Pod> {
        let mut results = Vec::new();

        let mut labels: BTreeMap<String, String> = BTreeMap::new();
        labels.insert("app".to_owned(), job_name.to_owned());

        if let Some(embedding_server) = &self.embeddingServer {
            let mut emb_server_spec: Vec<Pod> = (0..embedding_server.replicas)
                .into_iter()
                .map(|replica_idx| {
                    let mut podspec = self.gen_podspec_template();
                    let container = podspec
                        .containers
                        .first_mut()
                        .expect("no containers in a persia podspec template");

                    container.name = "emb-server".to_string();
                    container.args = Some(
                        vec![
                            "server",
                            "--embedding-config",
                            "$(EMBEDDING_CONFIG_PATH)",
                            "--global-config",
                            "$(GLOBAL_CONFIG_PATH)",
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

                    let env = container
                        .env
                        .as_mut()
                        .expect("no env in a persia podspec template");
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

                    Pod {
                        metadata: ObjectMeta {
                            name: Some(get_emb_server_pod_name(job_name, replica_idx)),
                            namespace: Some(namespace.to_owned()),
                            labels: Some(labels.clone()),
                            ..ObjectMeta::default()
                        },
                        spec: Some(podspec),
                        ..Pod::default()
                    }
                })
                .collect();

            results.append(&mut emb_server_spec);
        }

        if let Some(middleware_server) = &self.middlewareServer {
            let mut middleware_server_spec: Vec<Pod> = (0..middleware_server.replicas)
                .into_iter()
                .map(|replica_idx| {
                    let mut podspec = self.gen_podspec_template();
                    let container = &mut podspec
                        .containers
                        .first_mut()
                        .expect("no containers in a persia podspec template");

                    container.name = "middleware-server".to_string();
                    container.args = Some(
                        vec![
                            "middleware",
                            "--embedding-config",
                            "$(EMBEDDING_CONFIG_PATH)",
                            "--global-config",
                            "$(GLOBAL_CONFIG_PATH)",
                            "--replica-index",
                            "$(REPLICA_INDEX)",
                            "--replica-size",
                            "$(REPLICA_SIZE)",
                        ]
                        .into_iter()
                        .map(|x| x.to_string())
                        .collect(),
                    );

                    container.resources = middleware_server.resources.clone();
                    container.volume_mounts = middleware_server.volumeMounts.clone();

                    let env = container
                        .env
                        .as_mut()
                        .expect("no env in a persia podspec template");
                    env.push(EnvVar {
                        name: String::from("REPLICA_INDEX"),
                        value: Some(replica_idx.to_string()),
                        ..EnvVar::default()
                    });

                    env.push(EnvVar {
                        name: String::from("REPLICA_SIZE"),
                        value: Some(middleware_server.replicas.to_string()),
                        ..EnvVar::default()
                    });

                    if let Some(e) = &middleware_server.env {
                        e.iter().for_each(|env_var| {
                            env.push(env_var.clone());
                        });
                    }

                    Pod {
                        metadata: ObjectMeta {
                            name: Some(get_mid_server_pod_name(job_name, replica_idx)),
                            namespace: Some(namespace.to_owned()),
                            labels: Some(labels.clone()),
                            ..ObjectMeta::default()
                        },
                        spec: Some(podspec),
                        ..Pod::default()
                    }
                })
                .collect();

            results.append(&mut middleware_server_spec);
        }

        if let Some(trainer) = &self.trainer {
            let mut trainer_spec: Vec<Pod> = (0..trainer.replicas)
                .into_iter()
                .map(|replica_idx| {
                    let mut podspec = self.gen_podspec_template();
                    let container = &mut podspec
                        .containers
                        .first_mut()
                        .expect("no containers in a persia podspec template");

                    container.name = "trainer".to_string();
                    container.args = Some(
                        vec![
                            "trainer",
                            "$(TRAINER_PY_ENTRY_PATH)",
                            "--gpu-num",
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

                    container.resources = trainer.resources.clone();
                    container.volume_mounts = trainer.volumeMounts.clone();

                    let env = container
                        .env
                        .as_mut()
                        .expect("no env in a persia podspec template");
                    env.push(EnvVar {
                        name: String::from("REPLICA_INDEX"),
                        value: Some(replica_idx.to_string()),
                        ..EnvVar::default()
                    });
                    env.push(EnvVar {
                        name: String::from("REPLICA_SIZE"),
                        value: Some(trainer.replicas.to_string()),
                        ..EnvVar::default()
                    });
                    env.push(EnvVar {
                        name: String::from("NPROC_PER_NODE"),
                        value: Some(trainer.nprocPerNode.to_string()),
                        ..EnvVar::default()
                    });

                    if let Some(e) = &trainer.env {
                        e.iter().for_each(|env_var| {
                            env.push(env_var.clone());
                        });
                    }

                    Pod {
                        metadata: ObjectMeta {
                            name: Some(get_trainer_pod_name(job_name, replica_idx)),
                            namespace: Some(namespace.to_owned()),
                            labels: Some(labels.clone()),
                            ..ObjectMeta::default()
                        },
                        spec: Some(podspec),
                        ..Pod::default()
                    }
                })
                .collect();

            results.append(&mut trainer_spec);
        }

        if let Some(dataloader) = &self.dataloader {
            let mut dataloader_spec: Vec<Pod> = (0..dataloader.replicas)
                .into_iter()
                .map(|replica_idx| {
                    let mut podspec = self.gen_podspec_template();
                    let container = &mut podspec
                        .containers
                        .first_mut()
                        .expect("no containers in a persia podspec template");

                    container.name = "dataloader".to_string();
                    container.args = Some(
                        vec![
                            "compose",
                            "$(DATALOADER_PY_ENTRY_PATH)",
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

                    let env = container
                        .env
                        .as_mut()
                        .expect("no env in a persia podspec template");
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
                        name: String::from("DATALOADER_PY_ENTRY_PATH"),
                        value: self.dataLoaderPyEntryPath.clone(),
                        ..EnvVar::default()
                    });

                    if let Some(e) = &dataloader.env {
                        e.iter().for_each(|env_var| {
                            env.push(env_var.clone());
                        });
                    }

                    Pod {
                        metadata: ObjectMeta {
                            name: Some(get_dataloader_pod_name(job_name, replica_idx)),
                            namespace: Some(namespace.to_owned()),
                            labels: Some(labels.clone()),
                            ..ObjectMeta::default()
                        },
                        spec: Some(podspec),
                        ..Pod::default()
                    }
                })
                .collect();

            results.append(&mut dataloader_spec);
        }

        results
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
#[allow(non_snake_case)]
pub struct EmbeddingSpec {
    pub replicas: usize,
    pub resources: Option<ResourceRequirements>,
    pub volumeMounts: Option<Vec<VolumeMount>>,
    pub env: Option<Vec<EnvVar>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
#[allow(non_snake_case)]
pub struct MiddlewareSpec {
    pub replicas: usize,
    pub resources: Option<ResourceRequirements>,
    pub volumeMounts: Option<Vec<VolumeMount>>,
    pub env: Option<Vec<EnvVar>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
#[allow(non_snake_case)]
pub struct TrainerSpec {
    pub replicas: usize,
    pub nprocPerNode: usize,
    pub resources: Option<ResourceRequirements>,
    pub volumeMounts: Option<Vec<VolumeMount>>,
    pub env: Option<Vec<EnvVar>>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
#[allow(non_snake_case)]
pub struct DataLoaderSpec {
    pub replicas: usize,
    pub resources: Option<ResourceRequirements>,
    pub volumeMounts: Option<Vec<VolumeMount>>,
    pub env: Option<Vec<EnvVar>>,
}
