lint:
	pytype

flake8:
	python3 -m flake8 persia

format:
	python3 -m black --config pyproject.toml

pytest:
	pytest

all: lint flake8 format pytest

build_ci_image:
	docker build --build-arg DEPLOY=true --build-arg CI=true --build-arg HDFS=true \
		--no-cache -f docker/release/Dockerfile -t persiaml/persia-ci:latest .

build_cpu_image:
	docker build --build-arg BASE_IMAGE=ubuntu:20.04 --build-arg CPU_ONLY=cpuonly \
		--no-cache -f docker/release/Dockerfile -t persiaml/persia-cpu-runtime:latest .

build_cuda_image:
	docker build --no-cache -f docker/release/Dockerfile -t persiaml/persia-cuda-runtime:latest .

# default persia inference image is base on cuda runtime
build_inference_image:
	docker build --build-arg DEPLOY=true \
		--no-cache -f docker/release/Dockerfile -t persiaml/persia-inference-runtime:latest .

build_test_image:
	docker build --build-arg USE_CUDA=1 --no-cache -f docker/dev/Dockerfile -t persiaml-test:latest .

build_dev_image:
	docker build --build-arg USE_CUDA=1 --no-cache -f docker/dev/Dockerfile -t persiaml/persia-cuda-runtime:dev .

build_cuda_runtime_image:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda --no-cache -f docker/release/Dockerfile.multistage \
	-t persia-cuda-runtime --target runtime .

build_cuda_runtime_image_with_bagua:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda --no-cache -f docker/release/Dockerfile.multistage \
	-t persia-cuda-runtime:bagua --target bagua-runtime .

build_inference_runtime_image:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda --no-cache -f docker/release/Dockerfile.multistage \
	-t persia-inference-runtime --target inference-runtime.

build_cpu_runtime_image:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cpu --build-arg BASE_IMAGE="ubuntu:20.04" --no-cache -f docker/release/Dockerfile.multistage \
	-t persia-cpu-runtime --target runtime .

build_bagua_builder:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda --no-cache -f docker/release/Dockerfile.multistage \
	-t persia-cuda-runtime:test --target bagua-builder .
