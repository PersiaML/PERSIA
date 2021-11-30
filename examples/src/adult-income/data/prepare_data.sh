curl -o  train.csv https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -o test.csv https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
python3 data_preprocess.py --train-dataset train.csv --test-dataset test.csv --output_path .
rm test.csv train.csv