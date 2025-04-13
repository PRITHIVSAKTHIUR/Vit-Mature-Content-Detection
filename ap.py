import gradio as gr
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Vit-Mature-Content-Detection"  # Replace with your actual model path
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# Label mapping
labels = {
    "0": "Anime Picture",
    "1": "Hentai",
    "2": "Neutral",
    "3": "Pornography",
    "4": "Enticing or Sensual"
}

def mature_content_detection(image):
    """Predicts the type of content in the image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=mature_content_detection,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Vit-Mature-Content-Detection",
    description="Upload an image to classify whether it contains anime, hentai, neutral, pornographic, or enticing/sensual content."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
