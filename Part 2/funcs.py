# Auriel Wish
# Trial Project: Summer 2024 Internship
# ----------------------------------------------------------------
from transformers import pipeline

def eval_argument(arg):
    pipe = pipeline("text-classification", model="chkla/roberta-argument")
    # keep track of argument/argument quality pair
    argument = {}

    # run argument through the model
    result = pipe(arg)
    argument[arg] = round((result[0])['score'], 3)
    
    return argument