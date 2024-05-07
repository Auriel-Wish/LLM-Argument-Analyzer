# Auriel Wish
# Trial Project: Summer 2024 Internship
#
# funcs.py
# ----------------------------------------------------------------
from transformers import pipeline

def eval_argument(arg):
    pipe = pipeline("text-classification", model="chkla/roberta-argument")

    # run argument through the model
    # invert the score if NON_ARGUMENT value is greater than ARGUMENT value
    result = pipe(arg)
    if (result[0])['label'] == 'NON-ARGUMENT':
        (result[0])['score'] = 1 - (result[0])['score']
    if (result[0])['score'] < 0.01:
        (result[0])['score'] = 0
    return (round((result[0])['score'], 2))