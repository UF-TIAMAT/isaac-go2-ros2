import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class HFClipITC:
    def __init__(self, device=None, model_name="openai/clip-vit-large-patch14"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def cosine_image_text(self, image_np, txt: str) -> float:
        """Returns cosine similarity between image and text, like LAVIS `match_head='itc'`."""
        pil_img = Image.fromarray(image_np)

        # Processor handles both vision + text preprocessing
        inputs = self.processor(
            text=[txt],
            images=pil_img,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # Option A (recommended): true cosine over normalized embeddings
        img = outputs.image_embeds  # (1, D)
        txt = outputs.text_embeds   # (1, D)
        img = img / img.norm(p=2, dim=-1, keepdim=True)
        txt = txt / txt.norm(p=2, dim=-1, keepdim=True)
        cosine = (img @ txt.T).item()  # in [-1, 1]

        # Option B (alternate): CLIP’s built-in similarity logits (scaled cosine)
        # cosine = outputs.logits_per_image.squeeze().item() / self.model.logit_scale.exp().item()

        return cosine