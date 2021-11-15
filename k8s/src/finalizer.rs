use crate::crd::PersiaJob;
use kube::api::{Patch, PatchParams};
use kube::{Api, Client, Error};
use serde_json::{json, Value};

pub async fn add(client: Client, name: &str, namespace: &str) -> Result<PersiaJob, Error> {
    let api: Api<PersiaJob> = Api::namespaced(client, namespace);
    let finalizer: Value = json!({
        "metadata": {
            "finalizers": ["persiajobs.persia.com"]
        }
    });

    let patch: Patch<&Value> = Patch::Merge(&finalizer);
    Ok(api.patch(name, &PatchParams::default(), &patch).await?)
}

pub async fn delete(client: Client, name: &str, namespace: &str) -> Result<PersiaJob, Error> {
    let api: Api<PersiaJob> = Api::namespaced(client, namespace);
    let finalizer: Value = json!({
        "metadata": {
            "finalizers": null
        }
    });

    let patch: Patch<&Value> = Patch::Merge(&finalizer);
    Ok(api.patch(name, &PatchParams::default(), &patch).await?)
}
