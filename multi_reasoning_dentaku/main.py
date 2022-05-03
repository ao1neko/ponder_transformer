import argparse
from cmath import log
import numpy as np
import torch
import torch.optim as optim
from distutils.util import strtobool
import random
from torch.utils.data.dataset import Subset
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from method import train,valid
from transformers import BartTokenizer, BartForConditionalGeneration
from dentaku_tokenizer.tokenizer import BartDentakuTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer,Seq2SeqTrainingArguments
import numpy as np
from datasets import load_metric
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"




class SimpleDataset(Dataset):
    #TODO
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)        



def main(args):  
    tokenizer = BartDentakuTokenizer.from_pretrained("facebook/bart-base")
    tokenized_dataset = tokenizer(["A = 1, B = 12, C = A + B, C = ?","1","21"],padding=True, truncation=True, return_tensors='pt')
    train_dataset = SimpleDataset(tokenized_dataset,tokenized_dataset["input_ids"])
    valid_dataset = SimpleDataset(tokenized_dataset,tokenized_dataset["input_ids"])
    
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    training_args = Seq2SeqTrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_steps=3,
        learning_rate=0.0001,
        num_train_epochs=3,
        optim="adamw_torch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8, 
        seed=42,
        run_name="test/test",
        load_best_model_at_end=True,
        )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )
    
    trainer.train()
    output = trainer.evaluate(eval_dataset=valid_dataset)
    #print("metric 1",output)
    output = trainer.predict(test_dataset=valid_dataset)
    #print("metric 2",output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    args = parser.parse_args()
    main(args)
