# Auriel Wish
# Trial Project: Summer 2024 Internship
# ----------------------------------------------------------------

import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

filepath = 'arg_quality_rank_30k.csv'

# Load the saved model and tokenizer. This model was trained on a subset of the
# IBM data.
model_save_path = "Trained-Model"
model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)

# Create a pipeline for inference
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

def eval_argument(arg):
    # keep track of argument/argument quality pair
    argument = {}

    # run argument through the model
    result = pipe(arg)
    argument[arg] = round((result[0])['score'], 3)
    
    return argument

# evaluate arguments from IBM csv file
with open(filepath, 'r') as file:
    reader = csv.DictReader(file)
    for i, row_data in enumerate(reader):
        if (i == 0):
            continue
        if (i % 500 == 0):
            print(eval_argument(row_data['text']))
        if (i > 25000):
            break
