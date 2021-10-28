#!/bin/sh
set -ex

export IMAGE_TAG=tmp-${RANDOM:0:6}

upload_local_image() {
    remote_image_name=persiaml/$1:latest
    local_image_name=$1:$IMAGE_TAG
    docker tag $local_image_name $remote_image_name && docker push $remote_image_name && docker rmi $local_image_name $remote_image_name 
}

echo "current tmp image tag is " $IMAGE_TAG
make all_image -e

for image_name in "persia-ci" "persia-cuda-runtime" "persia-cpu-runtime"
do
    upload_local_image $image_name
done
