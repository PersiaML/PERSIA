IMAGE_TAG := test
DEVICE := cuda

lint:
	pytype

flake8:
	python3 -m flake8 persia

format:
	python3 -m black --config pyproject.toml

pytest:
	pytest

all: lint flake8 format

build_dev_pip:
	USE_CUDA=1 pip3 install -e . --prefix=~/.local/

build_ci_image:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda \
	-t persia-ci:$(IMAGE_TAG) --target builder .

build_dev_image:
	IMAGE_TAG=dev make build_cuda_runtime_image

build_cuda_runtime_image:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cuda \
	-t persia-cuda-runtime:$(IMAGE_TAG) --target runtime .

build_cpu_runtime_image:
	DOCKER_BUILDKIT=1 docker build --build-arg DEVICE=cpu --build-arg BASE_IMAGE="ubuntu:20.04" \
	-t persia-cpu-runtime:$(IMAGE_TAG) --target runtime .

build_runtime_image: build_cuda_runtime_image build_cpu_runtime_image

build_all_image: build_ci_image build_cuda_runtime_image build_cpu_runtime_image
