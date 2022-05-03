from transformers import BartTokenizer, BartForConditionalGeneration
from dentaku_tokenizer.tokenizer import BartDentakuTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartDentakuTokenizer.from_pretrained("facebook/bart-base")

inputs = tokenizer(["A=1, B = 12, C = A + B, C = ?","1","21"], padding=True, truncation=True, return_tensors='pt')
#print(inputs["input_ids"][0])

summary_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], num_beams=2, max_length=20, early_stopping=True)
summary_ids = summary_ids[:, 1:]

print([tokenizer.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=False) for g in summary_ids])
print([[tokenizer.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=False) for g in i] for i in inputs['input_ids']])



from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from dentaku_tokenizer.tokenizer import BartDentakuTokenizer

dataset = load_dataset("yelp_review_full")
tokenizer = BartDentakuTokenizer.from_pretrained("facebook/bart-base")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(
    seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(
    seed=42).select(range(10))

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")


metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="test_trainer", 
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=128, 
    per_device_eval_batch_size=128,
    learning_rate=0.0001,
    num_train_epochs=2,
    logging_strategy="epoch",
    seed=42,
    save_steps=500,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="test_trainer", 
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=128, 
    per_device_eval_batch_size=128,
    learning_rate=0.0001,
    num_train_epochs=2,
    logging_strategy="epoch",
    seed=42,
    save_steps=500,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
