# Auriel Wish
# Trial Project: Summer 2024 Internship
# ----------------------------------------------------------------

import csv
from transformers import pipeline

# Initialize roberta-argument model
pipe = pipeline("text-classification", model="chkla/roberta-argument")
filepath = 'IBM_Debater_(R)_arg_quality_rank_30k/arg_quality_rank_30k.csv'

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
        if (i > 50):
            break
        print(eval_argument(row_data['argument']))