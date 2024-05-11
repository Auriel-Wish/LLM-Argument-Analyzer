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

# Summarize the input if the argument is too long. This model best summaries
# passages that are less than 512 tokens.
# Limit number of words to 150 to try and limit response time for analysis.
def summarize(arg):
    pipe = pipeline("summarization", model="Falconsai/text_summarization")
    
    words = arg.split()
    result = arg
    was_changed = False
    if len(words) > 150:
        result = (pipe(arg, max_length = 150))[0]['summary_text']
        was_changed = True
        
    return [result, was_changed]

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

    # Not sure why we are not tokenizing but that's what the example said to do
    input = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # max_new_tokens: maximum number of words (technically tokens) that chatbot will generate
    # do_sample=True: don't automatically choose the word with the highest probability - allow for "creativity" 
    # top_k: only the 50 words with the highest probability can be chosen from
    # top_p: acknowledge the words with the highest probability such that their sum is 0.95/1
    # temperature: lower temperature leads to greater probability disparaty because the most likely words have higher probabilities than they would with higher temperature
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
            "content": "You are a chatbot who compares and analyzes arguments",
        },
        {"role": "user", "content": prompt}
    ]

    input = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = pipe(input, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    result = result[0]["generated_text"]
    result = result.split("|assistant|>\n")[1]

    return result