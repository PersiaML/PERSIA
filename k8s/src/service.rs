use crate::crd::PersiaJobSpec;
use crate::error::Error;
use crate::PersiaJobResources;

use kube::client::Client;

pub struct PersiaJobSchedulingService {
    pub kubernetes_client: Client,
}

impl PersiaJobSchedulingService {
    pub fn new(kubernetes_client: Client) -> Self {
        Self { kubernetes_client }
    }

    pub async fn apply(
        &self,
        job_name: &str,
        namespace: &str,
        json_spec: &str,
    ) -> Result<(), Error> {
        let spec: PersiaJobSpec = serde_json::from_str(json_spec)
            .map_err(|e| Error::JobSpecJsonDecodeError(format!("{:?}", e)))?;
        let job_resources =
            PersiaJobResources::new(&spec, job_name, namespace, self.kubernetes_client.clone());

        job_resources.apply().await?;

        Ok(())
    }

    pub async fn delete(&self, job_name: &str, namespace: &str) -> Result<(), Error> {
        PersiaJobResources::delete_services(self.kubernetes_client.clone(), namespace, job_name)
            .await?;
        PersiaJobResources::delete_pods(self.kubernetes_client.clone(), namespace, job_name)
            .await?;

        Ok(())
    }

    pub async fn list_pods(&self, job_name: &str, namespace: &str) -> Result<Vec<String>, Error> {
        let pods =
            PersiaJobResources::get_pods_name(self.kubernetes_client.clone(), namespace, job_name)
                .await?;
        Ok(pods)
    }

    pub async fn pod_log(&self, namespace: &str, pod_name: &str) -> Result<String, Error> {
        let log =
            PersiaJobResources::get_pod_log(self.kubernetes_client.clone(), namespace, pod_name)
                .await?;
        Ok(log)
    }

    pub async fn job_status(&self, namespace: &str, pod_name: &str) -> Result<String, Error> {
        let status =
            PersiaJobResources::get_pod_status(self.kubernetes_client.clone(), namespace, pod_name)
                .await?
                .ok_or(Error::NonePodStatusError)?;

        let status = serde_json::to_string(&status).unwrap();
        Ok(status)
    }
}
