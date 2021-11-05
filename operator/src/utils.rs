use k8s_openapi::api::apps::v1::{Deployment, DeploymentSpec};
use k8s_openapi::api::core::v1::{
    Container, ContainerPort, EnvVar, Pod, PodSpec, PodTemplateSpec, ResourceRequirements, Volume,
    VolumeMount,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::LabelSelector;
use k8s_openapi::Metadata;
use kube::api::{DeleteParams, ObjectMeta, PostParams};
use kube::{Api, Client, Error};
use std::collections::BTreeMap;

pub fn get_emb_server_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-emb-{}", job_name, replica_index)
}

pub fn get_mid_server_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-mid-{}", job_name, replica_index)
}

pub fn get_trainer_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-tr-{}", job_name, replica_index)
}

pub fn get_dataloader_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-dl-{}", job_name, replica_index)
}
