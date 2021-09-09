#!/bin/bash

python3 -m pip install -r docs/doc-requirements.txt
python3 -m sphinx_multiversion docs build/html
