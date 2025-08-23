# hf_blip2_itm_client.py
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForImageTextRetrieval

# Reuse your existing lightweight HTTP wrapper utils
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image


class HFBLIP2ITM:
    """
    HuggingFace BLIP-2 Image–Text Matching (ITM).

    Interface mirrors your previous LAVIS-based BLIP2ITM:
    - cosine(image: np.ndarray, txt: str) -> float
      Returns the probability of "match" from the ITM head (0..1).
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-itm-vit-g",
        device: Optional[Any] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if dtype is None:
            # FP16 on CUDA, FP32 on CPU for safety
            dtype = torch.float16 if (isinstance(device, torch.device) and device.type == "cuda") else torch.float32

        self.device = device
        self.dtype = dtype

        # Load model & processor
        self.model = Blip2ForImageTextRetrieval.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Returns ITM match probability in [0, 1] for (image, text).
        Kept as 'cosine' for API compatibility with your prior client.
        """
        pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image

        # Build tensors
        inputs = self.processor(images=pil_img, text=txt, return_tensors="pt")

        # Move to device and cast float tensors to model dtype
        moved = {}
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                moved[k] = v.to(self.device, dtype=self.dtype)
            else:
                moved[k] = v.to(self.device)

        # Use the ITM (classification) head
        out = self.model(**moved, use_image_text_matching_head=True)  # logits_per_image: [B, 2]
        probs = out.logits_per_image.softmax(dim=1)
        prob_match = probs[0, 1].item()  # probability of "match"

        return float(prob_match)


class HFBLIP2ITMClient:
    """
    Tiny HTTP client for the server below (same shape as your BLIP2ITMClient).
    """
    def __init__(self, port: int = 12183, route: str = "blip2itm_hf"):
        self.url = f"http://localhost:{port}/{route}"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12183)
    parser.add_argument("--route", type=str, default="blip2itm_hf")
    parser.add_argument("--model", type=str, default="Salesforce/blip2-itm-vit-g")
    args = parser.parse_args()

    print("Loading HF BLIP-2 ITM model...")

    class HFBLIP2ITMServer(ServerMixin, HFBLIP2ITM):
        def __init__(self, **kwargs):
            HFBLIP2ITM.__init__(self, **kwargs)
            ServerMixin.__init__(self)

        def process_payload(self, payload: dict) -> dict:
            # payload: {"image": <base64/bytes/str>, "txt": <str>}
            img = str_to_image(payload["image"])
            score = self.cosine(img, payload["txt"])
            return {"response": score}

    server = HFBLIP2ITMServer(model_name=args.model)
    print("Model loaded!")
    print(f"Hosting on http://localhost:{args.port}/{args.route}")
    host_model(server, name=args.route, port=args.port)