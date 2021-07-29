#!/bin/bash
set -x

mkdir /workspace/data_source -p && cd /workspace/data_source
curl -o  train.csv https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -o test.csv https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
cd /workspace && python3 data_preprocess.py