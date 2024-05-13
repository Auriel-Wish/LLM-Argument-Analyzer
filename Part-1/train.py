import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset
from optimum.bettertransformer import BetterTransformer

import time

fname = "arg_quality_rank_30k.csv"
out_dir = "Trained-Model"
# out_dir = "test_out_dir"

# use distilbert because it is smaller and faster than bert
# and is good for text classification
model_name = "distilbert-base-uncased"

# load in the dataset and get subset of it (3000 took 6 hours)
# split dataset to have both training data and eval data.
# Training data is used to train the model, eval data is used to test
# the model's performance
dataset = load_dataset('csv', data_files=fname, split='train')
train_set = dataset.shuffle(seed=42).select(range(5000))
train_set = train_set.train_test_split(test_size=0.1)

# tokenize the dataset (convert it to form that the model can use)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = train_set.map(tokenize_function)

data_collator = DataCollatorWithPadding(tokenizer)

# num_labels = 1 because the labels are continuous values [0, 1]
# The labels are the weighted average value provided by IBM
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Optimize training for CPU performance.
# model.to_bettertransformer()

training_args = TrainingArguments(
    output_dir=out_dir,
    # This would help if I knew it would be run on Intel processors
    # bf16=True,
    # use_ipex=True,
    # use_cpu=True,
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Setup the actual trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()
# model = BetterTransformer.reverse(model)

print(trainer.evaluate())
# trainer.push_to_hub("aurielwish")

# Save the model
model.save_pretrained('./' + out_dir)
trainer.tokenizer.save_pretrained('./' + out_dir)