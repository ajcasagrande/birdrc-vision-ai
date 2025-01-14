# Copyright (C) 2025 Anthony Casagrande
# AGPL-3.0 license

import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
from PIL import Image


# Define the CLIP Encoder
class CLIPEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.image_size = 224  # Default CLIP image size

    def preprocess(self, image, bbox):
        """
        Preprocess a bounding box image for CLIP.
        Args:
            image (numpy array): The full frame in BGR format.
            bbox (tuple): The bounding box coordinates (x1, y1, x2, y2).
        Returns:
            torch.Tensor: Preprocessed tensor for CLIP.
        """
        x, y, w, h = map(int, bbox[:4])
        cropped_image = image[y-h//2:y+h//2, x-w//2:x+w//2]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_image)
        inputs = self.processor(images=pil_image, return_tensors="pt", padding=True)
        return {key: value.to(self.device) for key, value in inputs.items()}

    def encode(self, image, bbox):
        """
        Encode a bounding box image to an embedding.
        Args:
            image (numpy array): The full frame in BGR format.
            bbox (tuple): The bounding box coordinates (x1, y1, x2, y2).
        Returns:
            numpy array: The CLIP embedding.
        """
        inputs = self.preprocess(image, bbox)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        return embeddings.cpu().numpy()
