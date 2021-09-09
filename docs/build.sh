#!/bin/bash

pip install -r docs/doc-requirements.txt
sphinx-multiversion docs build/html
