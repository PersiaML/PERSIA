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
