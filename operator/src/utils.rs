const DEFAULT_CUDA_IMAGE: String = String::from("persiaml/persia-cuda-runtime:latest");

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
