#!/bin/bash
set -x

cp ${BUILDKITE_BUILD_CHECKOUT_PATH}/k8s/resources/nats.operator.temp.yaml ${BUILDKITE_BUILD_CHECKOUT_PATH}/e2e/cache/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
sed -i 's/persia-nats-service/persia-nats-service-'${BUILDKITE_PIPELINE_ID}'/g' ${BUILDKITE_BUILD_CHECKOUT_PATH}/e2e/cache/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
kubectl apply -f ${BUILDKITE_BUILD_CHECKOUT_PATH}/e2e/cache/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml

docker run --rm -it -v $BUILDKITE_BUILD_CHECKOUT_PATH/e2e/cache:/cache persia-cpu-runtime:${BUILDKITE_PIPELINE_ID} bash -c "cp /opt/conda/lib/python3.8/site-packages/persia/e2e_test /cache"
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
$BUILDKITE_BUILD_CHECKOUT_PATH/e2e/cache/e2e_test;
result=$?;

kubectl delete -f ${BUILDKITE_BUILD_CHECKOUT_PATH}/e2e/cache/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml
rm ${BUILDKITE_BUILD_CHECKOUT_PATH}/e2e/cache/nats.operator.${BUILDKITE_PIPELINE_ID}.yaml

exit $result