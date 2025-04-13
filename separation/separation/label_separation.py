from datasets import load_dataset

# Load the dataset
dataset = load_dataset("YOUR-DATASET-HERE")

# Extract unique labels
labels = dataset["train"].features["label"].names

# Create id2label mapping
id2label = {str(i): label for i, label in enumerate(labels)}

# Print the mapping
print(id2label)
