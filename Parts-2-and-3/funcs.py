# Auriel Wish
# Trial Project: Summer 2024 Internship
#
# funcs.py
# ----------------------------------------------------------------
import torch
from transformers import pipeline, AutoTokenizer
# from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForSeq2SeqLM, ORTModelForCausalLM

# In theory, these functions would make the models run better on a CPU.
# I did not have time to figure this part and disect the warnings, so for now it is not implemented.

# def load_model_classification(model_name):
#     model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return [model, tokenizer]

# def load_model_summarization(model_name):
#     model = ORTModelForSeq2SeqLM.from_pretrained(model_name, export=True)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return [model, tokenizer]

# def load_model_generation(model_name):
#     model = ORTModelForCausalLM.from_pretrained(model_name, export=True)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return [model, tokenizer]

# evaluate the argument using a premade and pretrained HuggingFace model
def eval_argument_premade(arg):
    print("start eval")

    pipe = pipeline("text-classification", model="chkla/roberta-argument", return_all_scores=True)
    # pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    # run argument through the model
    result = (pipe(arg))[0]
    arg_val = -1
    for res in result:
        if res['label'] == 'ARGUMENT':
            arg_val = res['score']
            break

    if arg_val < 0.01:
        arg_val = 0
    
    print("end eval")
    return (round(arg_val, 2))

# evaluate the argument using a premade model that was also trained on the IBM dataset
def eval_argument_IBM(arg):
    print("start eval")

    pipe = pipeline("text-classification", model="aurielwish/trial-project", return_all_scores=True)
    # pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    # run argument through the model
    result = (pipe(arg))[0][0]
    arg_val = result['score']

    # This model seems to score all arguments between around 0.6 and 0.75
    # Scale the result to be equivalent but between 0 and 1.
    arg_val -= 0.6
    if arg_val < 0.01:
        arg_val = 0
    arg_val *= 9
    if (arg_val > 1):
        arg_val = 1

    print("end eval")
    return (round(arg_val, 2))

# Break down the components of the argument
def breakdown_argument(arg):
    print("start breakdown")
    
    pipe = pipeline("text-classification", model="raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L", return_all_scores=True)
    # pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    # run argument through the model
    result = (pipe(arg))[0]

    for res in result:
        res['score'] = round(res['score'], 2)
    
    print("end breakdown")
    return result

# Summarize the input if the argument is too long. This model best summaries
# passages that are less than 512 tokens.
# Limit number of words to 150 to try and limit response time for analysis.
def summarize(arg):
    print("start summarization")

    pipe = pipeline("summarization", model="Falconsai/text_summarization")
    # pipe = pipeline("summarization", model=model, tokenizer=tokenizer, return_all_scores=True)
    
    words = arg.split()
    result = arg
    was_changed = False
    if len(words) > 150:
        result = (pipe(arg, max_length = 150))[0]['summary_text']
        was_changed = True
        
    print("end summarization")
    return [result, was_changed]

# Get more detailed argument feedback from a chatbot
def get_feedback(arg, arg_score):
    print("start feedback")
    
    if arg_score < 0.5:
        good_or_bad = "bad"
    else:
        good_or_bad = "good"
    
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_all_scores=True)
    
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
    
    print("end feedback")
    return result

def compare(arg1, arg2):
    print("start compare")
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

    print("end compare")
    return result