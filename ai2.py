# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="WwQQ9F0aLwWflg7WGHrk"
)

# infer on a local image
result = CLIENT.infer("Ferrari.jpg", model_id="f1-car-2023/5")
