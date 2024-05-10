# Auriel Wish
# Trial Project: Summer 2024 Internship
#
# funcs.py
# ----------------------------------------------------------------
import torch
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

# Get more detailed argument feedback from a chatbot
def get_feedback(arg, arg_score):
    if arg_score < 0.5:
        good_or_bad = "bad"
    else:
        good_or_bad = "good"
    
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    
    prompt = "Why is \"" + arg + "\" a " + good_or_bad + " argument?"
    messages = [
        {
            "role": "system",
            "content": "You are a chatbot who evaluates arguments"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    input = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = pipe(input, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    result = result[0]["generated_text"]
    result = result.split("|assistant|>\n")[1]

    return result

def compare(arg1, arg2):
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    
    prompt = "Compare the arguments: \"" + arg1 + "\" and \"" + arg2 + "\""

    messages = [
        {
            "role": "system",
            "content": "You are a chatbot who compares 2 arguments against each other"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    input = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = pipe(input)

    result = result[0]["generated_text"]
    result = result.split("|assistant|>\n")[1]

    return result