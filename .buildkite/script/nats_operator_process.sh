#!/bin/bash
set -ex

if [[ $1 == "apply" ]]; then
    cp ${BUILDKITE_BUILD_CHECKOUT_PATH}/k8s/resources/nats.operator.temp.yaml ${BUILDKITE_BUILD_CHECKOUT_PATH}/e2e/cache/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
    sed -i 's/persia-nats-service/persia-nats-service-'${BUILDKITE_PIPELINE_ID}'/g' ${BUILDKITE_BUILD_CHECKOUT_PATH}/e2e/cache/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
    kubectl apply -f ${BUILDKITE_BUILD_CHECKOUT_PATH}/e2e/cache/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
elif [[ $1 == "delete" ]]; then
    kubectl delete -f ${BUILDKITE_BUILD_CHECKOUT_PATH}/e2e/cache/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
    rm ${BUILDKITE_BUILD_CHECKOUT_PATH}/e2e/cache/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
fi