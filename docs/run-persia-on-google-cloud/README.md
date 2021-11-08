Run Persia on Google Cloud
===

We assume that you already have a k8s cluster on Google Cloud, the following are the steps to deploy Persia to the k8s cluster.

1. Install NATS operator

[NATS Operator](https://github.com/nats-io/nats-operator) manages NATS clusters which is a dependency of Persia. You can install NATS operator with following command.

```bash
$ kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/00-prereqs.yaml
$ kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/10-deployment.yaml
```

2. Installing NVIDIA GPU device drivers

Also see Google Could [docs](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers).

```bash
$ kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/ubuntu/daemonset-preloaded.yaml
```

3. Run Persia

```bash
kubectl apply -f train.persia.yml
```

NOTE: The scripts and configuration files required for training are not in the docker image, they need to be accessible by the container through shared storage.
