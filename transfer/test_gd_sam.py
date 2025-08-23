from grounding_sam import GDSAMClient
from PIL import Image
import requests
import numpy as np  

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

client = GDSAMClient()
response = client.detections(image=np.array(image), target_prompt="a cat.")

print(response)
