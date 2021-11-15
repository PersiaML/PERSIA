#!/bin/bash
set -ex


upload_image() {
    docker tag $1 $2 
    docker push $2 
    docker rmi $2 # remove the remote image ref
}

remove_image() {
    docker rmi -f $1
}

for image_name in "persia-cuda-runtime" "persia-cpu-runtime"
do
    local_image_name=$image_name:$BUILDKITE_PIPELINE_ID
    remote_image_name=persiaml/$image_name:latest
    if [[ $1 == "upload" ]]; then
        upload_image $local_image_name $remote_image_name
    elif [[ $1 == "remove" ]]; then
        remove_image $local_image_name
    fi
done
