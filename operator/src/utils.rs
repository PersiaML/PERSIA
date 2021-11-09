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
