# Auriel Wish
# Trial Project: Summer 2024 Internship
# ----------------------------------------------------------------

import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load the saved model and tokenizer. This model was trained on a subset of the
# IBM data.
model_save_path = "Trained-Model"
model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)

# Create a pipeline for inference
classification_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example input
example_text = "the sky is blue so the sky is not blue"
result = classification_pipeline(example_text)

print(result)
