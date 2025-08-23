import torch
from PIL import Image
import requests
from transformers import AutoProcessor, Blip2ForImageTextRetrieval

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

model.to(device)
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# text = "two cats laying on a pink blanket"

image = Image.open("/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/results/August/08-18/rgb/frame_2025-08-18_07-51-04.png")
text = "a photo of a forklift"


inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.float16)
itm_out = model(**inputs, use_image_text_matching_head=True)
logits_per_image = torch.nn.functional.softmax(itm_out.logits_per_image, dim=1)
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities


print("Single prompt probabilities:")
print("Text prompt:", text)
print(f"{probs[0][0]:.1%} that image 0 is not '{text}'")

print(f"{probs[0][1]:.1%} that image 0 is '{text}'")

print("Multi-Class Softmax Probabilities:")
texts = ["a robot", "a forklift"]

inputs = processor(images=image, text=texts, return_tensors="pt", padding=True).to(device, torch.float16)
itc_out = model(**inputs, use_image_text_matching_head=False)
logits_per_image = itc_out.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")

print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")