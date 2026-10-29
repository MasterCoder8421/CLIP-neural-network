from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# Load the processor and model
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

image = Image.open("Photo on 27-10-26 at 11.35 AM.jpg")
texts = ["a photo of fried rice and beef", "a photo of fried rice and chicken", "a photo of fried rice, beef, and chicken" "a photo of noodles and beef", "a photo of noodles and chicken"]

inputs = processor(
    text=texts, images=image, return_tensors="pt", padding=True
)

# Forward pass to get similarity scores
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # image-text similarity score
probs = logits_per_image.softmax(dim=1)
print(probs)