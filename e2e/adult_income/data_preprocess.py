import os

from argparse import ArgumentParser

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder


def process(df_dataset: pd.core.frame.DataFrame, filepath: str):

    # categoricasl data encoded to int value
    for col in CATEGORICAL_COLUMNS:
        encoder = OrdinalEncoder()
        df_dataset[col] = encoder.fit_transform(df_dataset[[col]]).astype(np.uint64)

    categorical_data = np.vstack([df_dataset[k].values for k in CATEGORICAL_COLUMNS]).T
    continuous_data = np.vstack([df_dataset[k].values for k in CONTINUOUS_COLUMNS]).T

    df_dataset["target"] = df_dataset["income_bracket"].apply(lambda x: ">50K" in x)
    target = df_dataset["target"].to_numpy()

    np.savez_compressed(
        filepath,
        target=target.astype(np.float32),
        continuous_data=continuous_data.astype(np.float32).copy(),
        categorical_data=categorical_data,
        categorical_columns=CATEGORICAL_COLUMNS,
    )


if __name__ == "__main__":
    COLUMNS = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income_bracket",
    ]

    CATEGORICAL_COLUMNS = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native_country",
    ]

    CONTINUOUS_COLUMNS = [
        "age",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]

    parser = ArgumentParser()
    parser.add_argument("--train-dataset", default="data_source/train.csv")
    parser.add_argument("--test-dataset", default="data_source/test.csv")
    parser.add_argument("--output_path", default="data_source")
    args = parser.parse_args()

    output_path = args.output_path

    if args.train_dataset:
        df_train = pd.read_csv(args.train_dataset, names=COLUMNS, skipinitialspace=True)
        process(df_train, os.path.join(output_path, "train.npz"))

    if args.test_dataset:
        df_test = pd.read_csv(
            args.test_dataset, names=COLUMNS, skipinitialspace=True, skiprows=1
        )
        process(df_test, os.path.join(output_path, "test.npz"))
