import grpc
import os
import re
import sys
import json

sys.path.append("/workspace/proto/")
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import inference_pb2
import inference_pb2_grpc

from data_generator import make_dataloader
from persia.prelude import PyPersiaBatchData


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

    test_filepath = os.path.join("/data/", "test.npz")
    _, loader = make_dataloader(test_filepath, batch_size=1024)
    all_pred = []
    all_target = []

    for (dense, batch_sparse_ids, target) in tqdm(loader, desc="gen batch data..."):
        batch_data = PyPersiaBatchData()
        batch_data.add_dense([dense])
        batch_data.add_sparse(batch_sparse_ids, False)

        model_input = batch_data.to_bytes()
        prediction = infer(get_inference_stub(), "adult_income", model_input)

        assert len(prediction) == len(
            target
        ), f"miss results {len(prediction)} vs {len(target)}"

        all_target.append(target)
        all_pred.append(prediction)

    all_pred, all_target = np.concatenate(all_pred), np.concatenate(all_target)

    fpr, tpr, th = metrics.roc_curve(all_target, all_pred)
    infer_auc = metrics.auc(fpr, tpr)

    print(f"infer_auc = {infer_auc}")

    result_filepath = os.environ["RESULT_FILE_PATH"]
    with open(result_filepath, "r") as f:
        result = f.read()
        result = json.loads(result)

        eval_auc = result["eval_auc"]
        auc_diff = abs(eval_auc - infer_auc)
        assert (
            auc_diff < 1e-6
        ), f"infer error, expect auc diff < 1e-6 but got {auc_diff}"
