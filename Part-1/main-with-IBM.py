# Auriel Wish
# Trial Project: Summer 2024 Internship
# ----------------------------------------------------------------

import csv
from transformers import pipeline

pipe = pipeline("text-classification", model="aurielwish/trial-project")
filepath = 'arg_quality_rank_30k.csv'

def eval_argument(arg):
    # keep track of argument/argument quality pair
    argument = {}

    # run argument through the model
    result = pipe(arg)
    arg_val = (result[0])['score']

    # This model seems to score all arguments between around 0.6 and 0.75
    # Scale the result to be equivalent but between 0 and 1.
    arg_val -= 0.6
    if arg_val < 0.01:
        arg_val = 0
    arg_val *= 9
    if (arg_val > 1):
        arg_val = 1

    argument[arg] = round(arg_val, 2)
    
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