import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
import numpy as np


class HFBLIP2ImageTextRetrieval:
    def __init__(self, device=None, model_name="Salesforce/blip2-itm-vit-g"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            "Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

        self.model.to(self.device).eval()

    @torch.inference_mode()
    def cosine_image_text(self, image_np: np.ndarray, txt: str) -> float:
        image_pil = Image.fromarray(image_np)

        # Processor builds both vision + text tensors
        inputs = self.processor(images=image_pil, text=txt, return_tensors="pt").to(
            self.device, torch.float16
        )
        itm_out = self.model(**inputs, use_image_text_matching_head=True)
        logits_per_image = torch.nn.functional.softmax(itm_out.logits_per_image, dim=1)

        # probs[0][0]: Probability of no match, probs[0][1]: Probability of match
        prob = logits_per_image.softmax(dim=1)[0][1]

        return prob
