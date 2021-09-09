#!/bin/bash

set -euo pipefail

python3 -m pip install -r docs/doc-requirements.txt
git checkout self-hosted-docs
git pull --unshallow
python3 -m sphinx_multiversion docs build/html
# if [ "$BUILD_MULTIVERSION" == "1" ]; then
# 	git checkout main
# 	git pull --unshallow
# 	python3 -m sphinx_multiversion docs build/html
# else
# 	python3 -m sphinx docs build/html
# fi
# 
