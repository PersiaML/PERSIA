lint:
	pytype

flake8:
	python3 -m flake8 persia

format:
	python3 -m black --config pyproject.toml

pytest:
	pytest

all: lint flake8 format pytest