# Auriel Wish
# Trial Project: Summer 2024 Internship
# ----------------------------------------------------------------

import csv
from transformers import pipeline

# Initialize roberta-argument model. This model is pretrained specifically for
# the task of evaluating argument quality.
pipe = pipeline("text-classification", model="chkla/roberta-argument")
filepath = 'arg_quality_rank_30k.csv'

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