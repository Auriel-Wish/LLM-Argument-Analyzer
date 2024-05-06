from transformers import pipeline

# Initialize a text classification pipeline with the specific model
pipe = pipeline("text-classification", model="chkla/roberta-argument")

# Define some text to classify
input_text = "It has been determined that the amount of greenhouse gases have decreased by almost half because of the prevalence in the utilization of nuclear power."

# Classify the input text
classification_results = pipe(input_text)

# Display the classification results
for result in classification_results:
    label = result['label']  # The label (ARGUMENT or NON-ARGUMENT)
    score = result['score']  # The confidence score

print(f"Label: {label}, Score: {score:.4f}")
