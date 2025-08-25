from server_wrapper import send_request
from PIL import Image
import requests
import numpy as np  

class GDSAMClient:
    def __init__(self, port:int = 12183):
        self.url = f"http://localhost:{port}/gdsam"

    def detections(self, image: np.ndarray, target_prompt: str):
        print(f"GDSAMClient.detect_and_segment: {image.shape}, {target_prompt}" )
        response = send_request(self.url, image=image, target_prompt=target_prompt)
        return response

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image = Image.open("/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/transfer/detections/forklift./input_20250823_121424.png").convert("RGB")

client = GDSAMClient()
response = client.detections(image=np.array(image), target_prompt="a forklift.")

print(response)
