lint:
	pytype

flake8:
	python3 -m flake8 persia

format:
	python3 -m black --config pyproject.toml

pytest:
	pytest

all: lint flake8 format pytest

build_dev_image:
	docker build --build-arg USE_CUDA=1 --no-cache -t persiaml/persia-cuda-runtime:dev .