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
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda -f docker/release/Dockerfile.multistage \
	-t persia-ci:test --target builder .

build_dev_image:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda -f docker/release/Dockerfile.multistage \
	-t persia-ci:test --target builder .

build_cuda_runtime_image:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda -f docker/release/Dockerfile.multistage \
	-t persia-cuda-runtime:test --target runtime .

build_cuda_runtime_image_with_bagua:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda -f docker/release/Dockerfile.multistage \
	-t persia-cuda-runtime:test-bagua --target bagua-runtime .

build_inference_runtime_image:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda -f docker/release/Dockerfile.multistage \
	-t persia-inference-runtime:test --target inference-runtime .

build_cpu_runtime_image:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cpu --build-arg BASE_IMAGE="ubuntu:20.04" -f docker/release/Dockerfile.multistage \
	-t persia-cpu-runtime:test --target runtime .