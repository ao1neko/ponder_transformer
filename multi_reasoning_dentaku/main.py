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
from transformers import BartTokenizer, BartForConditionalGeneration
from dentaku_tokenizer.tokenizer import BartDentakuTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import os
from typing import List, Dict
from datasets2strlist import sample_datasets2strlist

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SimpleDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: torch.Tensor):
        """
        Args:
            encodings (Dict[torch.Tensor]): tokenized input 
                {'input_ids': tensor([
                    [    0,    83,  5457,   112,  2156,   163,  5457,   112,   176,  2156,
                        230,  5457,    83,  2055,   163,  2156,   230,  5457, 17487,     2],
                    [    0,   112,     2,     1,     1,     1,     1,     1,     1,     1,
                        1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
                    [    0,   132,   134,     2,     1,     1,     1,     1,     1,     1,
                        1,     1,     1,     1,     1,     1,     1,     1,     1,     1]]), 
                'attention_mask': tensor([
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
            labels (torch.Tensor): tokenized output
                tensor([
                    [    0,    83,  5457,   112,  2156,   163,  5457,   112,   176,  2156,
                        230,  5457,    83,  2055,   163,  2156,   230,  5457, 17487,     2],
                    [    0,   112,     2,     1,     1,     1,     1,     1,     1,     1,
                        1,     1,     1,     1,     1,     1,     1,     1,     1,     1],
                    [    0,   132,   134,     2,     1,     1,     1,     1,     1,     1,
                        1,     1,     1,     1,     1,     1,     1,     1,     1,     1]])
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def assign_GPU(tokenized_inputs: Dict[str, torch.Tensor], device: torch.device = torch.device("cuda:0")):
    assign_GPU_tokenized_inputs = {}
    for key, val in tokenized_inputs.items():
        assign_GPU_tokenized_inputs[key] = val.to(device)
    return assign_GPU_tokenized_inputs


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    datasets_name = args.datasets_name
    tokenizer = BartDentakuTokenizer.from_pretrained("facebook/bart-base")


    # inputs,labels：List[str]
    assert datasets_name in ["sample"], "datasets_nameが間違っています"
    if datasets_name == "sample":
        inputs,labels = sample_datasets2strlist()
    tokenized_inputs = assign_GPU(
        tokenizer(inputs, padding=True, truncation=True, return_tensors='pt'), device)
    tokenized_labels = assign_GPU(
        tokenizer(labels, padding=True, truncation=True, return_tensors='pt'), device)
    train_dataset = SimpleDataset(tokenized_inputs, tokenized_labels["input_ids"])
    valid_dataset = SimpleDataset(tokenized_inputs, tokenized_labels["input_ids"])
    
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-base").to(device)
    training_args = Seq2SeqTrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_steps=3,
        learning_rate=0.0001,
        num_train_epochs=1,
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
    #output = trainer.evaluate(eval_dataset=valid_dataset)
    #print("metric 1",output)
    #output = trainer.predict(test_dataset=valid_dataset)
    #print("metric 2",output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--datasets_name', default="sample")
    args = parser.parse_args()
    main(args)
