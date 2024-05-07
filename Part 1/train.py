from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset

fname = "arg_quality_rank_30k.csv"
# use distilbert because it is smaller and faster than bert (another model)
model_name = "distilbert-base-uncased"

# load in the dataset and get subset of it (3000 took multiple hours)
dataset = load_dataset('csv', data_files=fname, split='train')
train_set = dataset.shuffle(seed=42).select(range(10))
train_set = train_set.train_test_split(test_size=0.1)

print(train_set)

# tokenize the dataset (convert it to form that the model can use)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = train_set.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

# num_labels = 1 because the labels are continuous values [0, 1]
# The labels are the weighted average value provided by IBM
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
training_args = TrainingArguments(output_dir="Trained-Model")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer
)
trainer.train()

model.save_pretrained('./Trained-Model')
trainer.tokenizer.save_pretrained('./Trained-Model')