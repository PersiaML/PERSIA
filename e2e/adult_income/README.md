# Launch Persia Single Machine Example
1. Download the Adult Census Income first. [uci website](https://archive.ics.uci.edu/ml/datasets/Adult)
    ```bash
    mkdir examples/SingleMachine/data_source -p && cd examples/SingleMachine/data_source
    curl -o  train.csv https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    curl -o test.csv https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
    cd .. && python3 data_preprocess.py
    
    ```
2. Pull required docker images
    ```bash
    docker pull persiaml/persia-cpu-runtime:latest
    docker pull persiaml/persia-gpu-runtime:latest
    ```
3. Preprocess raw data
    ```bash
    make process_data
    ```
4. Start training
    ```bash
    make run
    ```
