pub mod crd;
pub mod error;
pub mod finalizer;
pub mod service;

use crate::crd::{PersiaJob, PersiaJobSpec};
use k8s_openapi::api::core::v1::{Pod, PodStatus, Service};
use kube::api::{DeleteParams, ListParams, LogParams, PostParams};
use kube::{Api, Client, Error};

pub struct PersiaJobResources {
    pub job_name: String,
    pub namespace: String,
    pub pods: Vec<Pod>,
    pub services: Vec<Service>,
    pub kubernetes_client: Client,
}

impl PersiaJobResources {
    pub fn new(
        spec: &PersiaJobSpec,
        job_name: &str,
        namespace: &str,
        kubernetes_client: Client,
    ) -> Self {
        let pods = spec.gen_pods(job_name, namespace);
        let services = spec.gen_services(job_name, namespace);
        Self {
            job_name: job_name.to_owned(),
            namespace: namespace.to_owned(),
            pods,
            services,
            kubernetes_client,
        }
    }

    pub async fn delete(&self) -> Result<(), Error> {
        Self::delete_services(
            self.kubernetes_client.clone(),
            &self.namespace,
            &self.job_name,
        )
        .await?;
        Self::delete_pods(
            self.kubernetes_client.clone(),
            &self.namespace,
            &self.job_name,
        )
        .await?;
        Ok(())
    }

    pub async fn delete_resources(
        kubernetes_client: Client,
        namespace: &str,
        job_name: &str,
    ) -> Result<(), Error> {
        Self::delete_services(kubernetes_client.clone(), namespace, job_name).await?;
        Self::delete_pods(kubernetes_client.clone(), namespace, job_name).await?;
        Ok(())
    }

    pub async fn apply(&self) -> Result<(), Error> {
        self.deploy_pods().await?;
        self.deploy_services().await?;
        Ok(())
    }

    pub async fn deploy_pods(&self) -> Result<Vec<Pod>, Error> {
        let pod_api: Api<Pod> = Api::namespaced(self.kubernetes_client.clone(), &self.namespace);
        let pp = PostParams::default();
        let futs: Vec<_> = self.pods.iter().map(|p| pod_api.create(&pp, p)).collect();

        let result = futures::future::try_join_all(futs).await;
        result
    }

    pub async fn deploy_services(&self) -> Result<Vec<Service>, Error> {
        let service_api: Api<Service> =
            Api::namespaced(self.kubernetes_client.clone(), &self.namespace);
        let pp = PostParams::default();
        let futs: Vec<_> = self
            .services
            .iter()
            .map(|p| service_api.create(&pp, p))
            .collect();

        let result = futures::future::try_join_all(futs).await;
        result
    }

    pub async fn delete_services(
        kubernetes_client: Client,
        namespace: &str,
        job_name: &str,
    ) -> Result<(), Error> {
        let service_api: Api<Service> = Api::namespaced(kubernetes_client, namespace);
        let label_selector = crd::get_label_selector(job_name);
        let lp = ListParams::default().labels(label_selector.as_str());
        let services = service_api.list(&lp).await?;

        let mut services_name = Vec::new();
        services.iter().for_each(|s| {
            if let Some(service_name) = &s.metadata.name {
                services_name.push(service_name.clone());
            }
        });

        if services_name.len() == 0 {
            return Ok(());
        }

        let dp = DeleteParams::default();
        let futs: Vec<_> = services_name
            .iter()
            .map(|p| service_api.delete(p, &dp))
            .collect();

        let result = futures::future::try_join_all(futs).await.map(|_| ());
        result
    }

    pub async fn get_jobs_name(
        kubernetes_client: Client,
        namespace: &str,
    ) -> Result<Vec<String>, Error> {
        let job_api: Api<PersiaJob> = Api::namespaced(kubernetes_client, namespace);
        let lp = ListParams::default();
        let jobs = job_api.list(&lp).await?;

        let mut jobs_name: Vec<String> = Vec::new();
        jobs.iter().for_each(|j| {
            if let Some(job_name) = &j.metadata.name {
                jobs_name.push(job_name.clone());
            }
        });

        Ok(jobs_name)
    }

    pub async fn get_pods_name(
        kubernetes_client: Client,
        namespace: &str,
        job_name: &str,
    ) -> Result<Vec<String>, Error> {
        let pod_api: Api<Pod> = Api::namespaced(kubernetes_client, namespace);
        let label_selector = crd::get_label_selector(job_name);
        let lp = ListParams::default().labels(label_selector.as_str());
        let pods = pod_api.list(&lp).await?;

        let mut pods_name = Vec::new();
        pods.iter().for_each(|p| {
            if let Some(pod_name) = &p.metadata.name {
                pods_name.push(pod_name.clone());
            }
        });

        Ok(pods_name)
    }

    pub async fn delete_pods(
        kubernetes_client: Client,
        namespace: &str,
        job_name: &str,
    ) -> Result<(), Error> {
        // let label_selector = crd::get_label_selector(job_name);
        // let lp = ListParams::default().labels(label_selector.as_str());
        // let dp = DeleteParams::default();

        // let pod_api: Api<Pod> = Api::namespaced(kubernetes_client, namespace);
        // let result = pod_api.delete_collection(&dp, &lp).await?;

        let pods_name = Self::get_pods_name(kubernetes_client.clone(), namespace, job_name).await?;
        if pods_name.len() == 0 {
            return Ok(());
        }

        let pod_api: Api<Pod> = Api::namespaced(kubernetes_client, namespace);
        let dp = DeleteParams::default();
        let futs: Vec<_> = pods_name.iter().map(|p| pod_api.delete(p, &dp)).collect();

        let result = futures::future::try_join_all(futs).await.map(|_| ());
        result
    }

    pub async fn get_pod_status(
        kubernetes_client: Client,
        namespace: &str,
        pod_name: &str,
    ) -> Result<Option<PodStatus>, Error> {
        let pod_api: Api<Pod> = Api::namespaced(kubernetes_client, namespace);
        let pod = pod_api.get_status(pod_name).await?;

        Ok(pod.status)
    }

    pub async fn get_pod_log(
        kubernetes_client: Client,
        namespace: &str,
        pod_name: &str,
    ) -> Result<String, Error> {
        let pod_api: Api<Pod> = Api::namespaced(kubernetes_client, namespace);
        let lp = LogParams::default();
        let log = pod_api.logs(pod_name, &lp).await?;
        Ok(log)
    }
}
