use k8s_openapi::api::core::v1::{Pod, Service};
use kube::api::{DeleteParams, PostParams};
use kube::{Api, Client, Error};

pub async fn deploy_pods(
    client: Client,
    pods: &Vec<Pod>,
    namespace: &str,
) -> Result<Vec<Pod>, Error> {
    let pod_api: Api<Pod> = Api::namespaced(client, namespace);
    let pp = PostParams::default();
    let futs: Vec<_> = pods.iter().map(|p| pod_api.create(&pp, p)).collect();

    let result = futures::future::try_join_all(futs).await;
    result
}

pub async fn delete_pods(
    client: Client,
    pods_name: &Vec<String>,
    namespace: &str,
) -> Result<(), Error> {
    let pod_api: Api<Pod> = Api::namespaced(client, namespace);
    let dp = DeleteParams::default();
    let futs: Vec<_> = pods_name.iter().map(|p| pod_api.delete(p, &dp)).collect();

    let result = futures::future::try_join_all(futs).await.map(|_| ());
    result
}

pub async fn deploy_services(
    client: Client,
    services: &Vec<Service>,
    namespace: &str,
) -> Result<Vec<Service>, Error> {
    let service_api: Api<Service> = Api::namespaced(client, namespace);
    let pp = PostParams::default();
    let futs: Vec<_> = services
        .iter()
        .map(|p| service_api.create(&pp, p))
        .collect();

    let result = futures::future::try_join_all(futs).await;
    result
}

pub async fn delete_services(
    client: Client,
    services_name: &Vec<String>,
    namespace: &str,
) -> Result<(), Error> {
    let service_api: Api<Service> = Api::namespaced(client, namespace);
    let dp = DeleteParams::default();
    let futs: Vec<_> = services_name
        .iter()
        .map(|p| service_api.delete(p, &dp))
        .collect();

    let result = futures::future::try_join_all(futs).await.map(|_| ());
    result
}

pub const DEFAULT_CUDA_IMAGE: &str = "persiaml/persia-cuda-runtime:latest";

pub fn get_emb_server_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-embedding-server-{}", job_name, replica_index)
}

pub fn get_mid_server_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-middware-server-{}", job_name, replica_index)
}

pub fn get_trainer_pod_name(job_name: &str, replica_index: usize) -> String {
    format!("{}-trainer-{}", job_name, replica_index)
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
