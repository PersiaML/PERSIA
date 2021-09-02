import grpc
import sys
sys.path.append('/workspace/proto/')

import inference_pb2
import inference_pb2_grpc

import numpy as np

from persia.prelude import PyPersiaBatchData

def get_inference_stub():
    channel = grpc.insecure_channel('localhost:7070')
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub

def infer(stub, model_name, model_input):
    input_data = {'batch': model_input}
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data))

    try:
        prediction = response.prediction.decode('utf-8')
        print(prediction)
    except grpc.RpcError as e:
        exit(1)

if __name__ == '__main__':

    batch_size = 128
    feature_dim = 8
    denses = [np.random.rand(batch_size, 5).astype(np.float32)]
    sparse = []
    for sparse_idx in range(8):
        sparse.append((
            f'feature{sparse_idx + 1}',
            [np.random.randint(1000000, size=feature_dim).astype(np.uint64) for _ in range(batch_size)]
        ))
    batch_data = PyPersiaBatchData()
    batch_data.add_dense(denses)
    batch_data.add_sparse(sparse)
    model_input = batch_data.to_bytes()
    infer(get_inference_stub(), 'adult_income', model_input)
