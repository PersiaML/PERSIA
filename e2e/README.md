# Persia CI Process

## Build Multiple Persia Runtime Image

The first step is to build the persia runtime image for e2e ci test.There will build two type of runtime to cover the cpu and cuda environment with `$BUILDKITE_PIPELINE_ID` tag.

*ci runtime image below*
- persia-cpu-runtime:$BUILDKITE_PIPELINE_ID
- persia-cuda-runtime:$BUILDKITE_PIPELINE_ID


## Provide The Resource Folder
To prevent the `buildkite-agent` can't remove the own files after docker-compose exit, there provide the `Persia/e2e/resource` folder to place the shared files when running multiple times docker-compose within one step.
