import grpc
import os
import sys
import json

sys.path.append("/cache/proto/")

import numpy as np

from tqdm import tqdm
from sklearn import metrics
from persia.embedding.data import PersiaBatch

import inference_pb2
import inference_pb2_grpc

from data_generator import make_dataloader


def get_inference_stub():
    channel = grpc.insecure_channel("localhost:7070")
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub


def infer(stub, model_name, model_input):
    input_data = {"batch": model_input}
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data)
    )
    try:
        prediction = response.prediction.decode("utf-8")
        prediction = prediction.splitlines()
        prediction = [x.strip() for x in prediction]
        prediction = [x.replace(",", "") for x in prediction]
        prediction = prediction[1:-1]
        prediction = [float(x) for x in prediction]
        return prediction
    except:
        exit(1)


if __name__ == "__main__":

    test_filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data/test.npz"
    )
    loader = make_dataloader(test_filepath, batch_size=1024)
    all_pred = []
    all_label = []

    for (non_id_type_feature, id_type_features, label) in tqdm(
        loader, desc="gen batch data..."
    ):
        batch_data = PersiaBatch(
            id_type_features,
            non_id_type_features=[non_id_type_feature],
            requires_grad=False,
        )
        model_input = batch_data.to_bytes()
        prediction = infer(get_inference_stub(), "adult_income", model_input)

        assert len(prediction) == len(
            label
        ), f"Missing results: prediction length({len(prediction)}) does not match label length({len(label)})"

        all_label.append(label.data)
        all_pred.append(prediction)

    all_pred, all_label = np.concatenate(all_pred), np.concatenate(all_label)

    fpr, tpr, th = metrics.roc_curve(all_label, all_pred)
    infer_auc = metrics.auc(fpr, tpr)

    print(f"infer_auc = {infer_auc}")

    assert (
        infer_auc > 0.8927
    ), f"infer error, expect infer_auc > 0.8927 but got {infer_auc}"
