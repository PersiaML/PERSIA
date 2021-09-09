#!/bin/bash

set -euo pipefail

git pull --unshallow
python3 -m pip install -r docs/doc-requirements.txt
if [ "$BUILD_MULTIVERSION" == "1" ]; then
	python3 -m sphinx_multiversion docs build/html
else
	python3 -m sphinx docs build/html
fi

