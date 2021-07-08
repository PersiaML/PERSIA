format:
	python3 -m black persia --config pyproject.toml

lint: 
	pytype persia --config setup.cfg
