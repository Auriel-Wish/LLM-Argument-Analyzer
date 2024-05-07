# Auriel Wish
# Trial Project: Summer 2024 Internship
#
# funcs.py
# ----------------------------------------------------------------
from transformers import pipeline

def eval_argument(arg):
    pipe = pipeline("text-classification", model="chkla/roberta-argument", return_all_scores=True)

    # run argument through the model
    result = (pipe(arg))[0]
    arg_val = -1
    for res in result:
        if res['label'] == 'ARGUMENT':
            arg_val = res['score']
            break

    if arg_val < 0.01:
        arg_val = 0
    return (round(arg_val, 2))