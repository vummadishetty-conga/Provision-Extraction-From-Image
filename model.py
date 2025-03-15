import torch
from transformers import CLIPProcessor, CLIPModel

import os

# Specify a local directory to store the model
def initiate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = os.path.join("C:\\Users\\vummadisetty\\OneDrive - Conga\\Desktop\\ProvisionExtractionfromImage\\", ".cache", "clip_model")
    print(cache_dir)

    # Load CLIP Model, specifying the cache directory
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        device_map=device,
        cache_dir=cache_dir,  # Store the model here
    )
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14",
        cache_dir=cache_dir,  # Store the processor here
    )
    return model, processor, device


# def initiate_model():
#     #model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     model = CLIPModel.from_pretrained(
#         "openai/clip-vit-base-patch32",
#         device_map=device,
#     )
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     return model,processor,device