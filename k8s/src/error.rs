#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Kubernetes reported error: {source}")]
    KubeError {
        #[from]
        source: kube::Error,
    },
    #[error("Invalid PersiaJob CRD: {0}")]
    UserInputError(String),
    #[error("Failed to decode json format PersiaJobSpec: {0}")]
    JobSpecJsonDecodeError(String),
    #[error("Pod status is None")]
    NonePodStatusError,
}
