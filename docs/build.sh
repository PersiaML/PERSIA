#!/bin/bash

set -euo pipefail

python3 -m pip install -r docs/doc-requirements.txt
if [ "$BUILD_MULTIVERSION" == "1" ]; then
	git fetch --all --tags
	git checkout main
	git pull --unshallow
	python3 -m sphinx_multiversion docs build/html
	cp docs/redirect-index.html build/html/index.html
else
	python3 -m sphinx docs build/html
fi
