#!/bin/bash
set -ex

if [[ $1 == "apply" ]]; then
    cp k8s/resources/nats.operator.temp.yaml k8s/resources/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
    sed -i 's/persia-nats-service/persia-nats-service-'${BUILDKITE_PIPELINE_ID}'/g' k8s/resources/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
    kubectl apply -f k8s/resources/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
elif [[ $1 == "delete" ]]; then
    kubectl delete -f k8s/resources/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
    rm k8s/resources/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
fi