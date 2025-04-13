![uijytyyt.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/IDbH_a4KQpydEVQ4VtKiA.png)
 
# **Vit-Mature-Content-Detection**

> **Vit-Mature-Content-Detection** is an image classification vision-language model fine-tuned from **vit-base-patch16-224-in21k** for a single-label classification task. It classifies images into various mature or neutral content categories using the **ViTForImageClassification** architecture.

> [!Note]
> Use this model to support positive, safe, and respectful digital spaces. Misuse is strongly discouraged and may violate platform or regional policies. This model doesn't generate any unsafe content, as it is a classification model and does not fall under the category of models not suitable for all audiences.

> [!Important]
> Neutral = Safe / Normal

```py
Classification Report:
                     precision    recall  f1-score   support

      Anime Picture     0.9311    0.9455    0.9382      5600
             Hentai     0.9520    0.9244    0.9380      4180
            Neutral     0.9681    0.9529    0.9604      5503
        Pornography     0.9896    0.9832    0.9864      5600
Enticing or Sensual     0.9602    0.9870    0.9734      5600

           accuracy                         0.9605     26483
          macro avg     0.9602    0.9586    0.9593     26483
       weighted avg     0.9606    0.9605    0.9604     26483
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/FvFTPm_JKwFIffb_LF4ft.png)

```py
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("YOUR-DATASET-HERE")

# Extract unique labels
labels = dataset["train"].features["label"].names

# Create id2label mapping
id2label = {str(i): label for i, label in enumerate(labels)}

# Print the mapping
print(id2label)
```

---

The model categorizes images into five classes:

- **Class 0:** Anime Picture  
- **Class 1:** Hentai  
- **Class 2:** Neutral  
- **Class 3:** Pornography  
- **Class 4:** Enticing or Sensual 

# **Run with Transformers 🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```

---

# **Recommended Use Cases**
- Content moderation systems
- Parental control filters
- Dataset preprocessing and filtering
- Digital well-being and user safety tools
- Search engine safe filter enhancements

# **Discouraged / Prohibited Use**
- Harassment or shaming
- Unethical surveillance
- Illegal or deceptive applications
- Sole-dependency without human oversight
- Misuse to mislead moderation decisions
