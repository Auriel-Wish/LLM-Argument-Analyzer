# Auriel Wish
# Trial Project: Summer 2024 Internship
#
# funcs.py
# ----------------------------------------------------------------
from transformers import pipeline

# evaluate the argument using a premade and pretrained HuggingFace model
def eval_argument_premade(arg):
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

# evaluate the argument using a premade model that was also trained on the IBM dataset
def eval_argument_IBM(arg):
    pipe = pipeline("text-classification", model="aurielwish/trial-project", return_all_scores=True)

    # run argument through the model
    result = (pipe(arg))[0][0]
    arg_val = result['score']

    if arg_val < 0.01:
        arg_val = 0
    return (round(arg_val, 2))

# Break down the components of the argument
def breakdown_argument(arg):
    pipe = pipeline("text-classification", model="raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L", return_all_scores=True)

    # run argument through the model
    result = (pipe(arg))[0]

    for res in result:
        res['score'] = round(res['score'], 2)

    return result

# Summarize the input if the argument is too long. Summarize the summary
# if that is also too long, and continue to do so until the argument length
# can be inputted into the classification models.
def summarize(arg):
    pipe = pipeline("summarization", model="Falconsai/text_summarization")
    
    words = arg.split()
    result = arg
    while len(words) > 450:
        result = (pipe(result))[0]['summary_text']
        words = result.split()
        
    return result
