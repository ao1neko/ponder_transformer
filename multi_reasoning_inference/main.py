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
from transformers import BartTokenizer,BartConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration
from bart.modeling_bart import BartForConditionalGeneration
from reasoning_bart.modeling_reasoning_bart import ReasoningBartForConditionalGeneration
from dentaku_tokenizer.t5_tokenizer import T5DentakuTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import os
from typing import List, Dict

from utils import read_jsonl_file,read_jsonl_files,at_once_predict,iterative_predict,clean_predict_text,eliminate_calculated_index

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




def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    architecture_name = args.architecture_name
    train_dataset_names = args.train_dataset_names.split(",")
    valid_dataset_names = args.valid_dataset_names.split(",")
    test_dataset_names = args.test_dataset_names.split(",")
    load_model_dir = args.load_model_dir
    model_name = args.model_name
    train= strtobool(args.train)
    predict = strtobool(args.predict)
    train_epochs = args.train_epochs
    eval_steps = args.eval_steps
    output_dir = args.output_dir
    run_dir = args.run_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    assert architecture_name in ["at_once","iterative","ansonly"], "architecture_nameが間違っています"
    assert model_name in ["bart","reasoning_bart","t5"],"model_nameが間違っています"
    
    if model_name in ["bart","reasoning_bart"]:
        tokenizer = BartDentakuTokenizer.from_pretrained("facebook/bart-base")
    elif model_name in ["t5"]:
        tokenizer = T5DentakuTokenizer.from_pretrained("t5-base")
        tokenizer_test = T5Tokenizer.from_pretrained("t5-base")
     
    # inputs,labels：List[str]    
    train_inputs, train_labels = read_jsonl_files(train_dataset_names)
    valid_inputs, valid_labels = read_jsonl_files(valid_dataset_names)
                
    train_tokenized_inputs = tokenizer(
        train_inputs, padding=True, truncation=True, return_tensors='pt')
    train_tokenized_labels = tokenizer(
        train_labels, padding=True, truncation=True, return_tensors='pt')
    valid_tokenized_inputs = tokenizer(
        valid_inputs, padding=True, truncation=True, return_tensors='pt')
    valid_tokenized_labels = tokenizer(
        valid_labels, padding=True, truncation=True, return_tensors='pt')
    
    
    ids = tokenizer("text","text")["input_ids"]
    print(ids)
    print(train_inputs[0])
    print("\n")
    
    print([tokenizer.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in ids])
    print([tokenizer.decode([0,1,2], skip_special_tokens=False, clean_up_tokenization_spaces=False)])
    print([tokenizer.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in train_tokenized_inputs.input_ids[0]])
    print("\n")
    
    print([tokenizer_test.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in train_tokenized_inputs.input_ids[0]])
    print([tokenizer.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in train_tokenized_inputs.input_ids[0]])
    print([tokenizer_test.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in train_tokenized_inputs.input_ids[1]])
    print([tokenizer.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in train_tokenized_inputs.input_ids[1]])
    
    exit()
    
 
    train_dataset = SimpleDataset(
        train_tokenized_inputs, train_tokenized_labels["input_ids"])
    valid_dataset = SimpleDataset(
        valid_tokenized_inputs, valid_tokenized_labels["input_ids"])
    

    
    if model_name == "bart":
        model = BartForConditionalGeneration.from_pretrained(
            load_model_dir).to(device)
    elif model_name == "reasoning_bart":
        config = BartConfig().from_pretrained(load_model_dir)
        config.output_hidden_states = True
        model = ReasoningBartForConditionalGeneration.from_pretrained(
            load_model_dir,config=config).to(device)
    elif model_name == "t5":
        model = T5ForConditionalGeneration.from_pretrained(load_model_dir).to(device)

    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        learning_rate=learning_rate,
        num_train_epochs=train_epochs,
        optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=42,
        run_name=run_dir,
        load_best_model_at_end=True,
        predict_with_generate=True
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    if train:
        trainer.train()
    if predict:
        if architecture_name in ["ansonly","at_once"]:
            test_inputs, test_labels = read_jsonl_files(test_dataset_names)
            test_tokenized_inputs = tokenizer(test_inputs, padding=True, truncation=True, return_tensors='pt')
            test_tokenized_labels = tokenizer(test_labels, padding=True, truncation=True, return_tensors='pt')
            test_dataset = SimpleDataset(test_tokenized_inputs, test_tokenized_labels["input_ids"])
            predict_id_list,label_id_list, metrics = trainer.predict(test_dataset=test_dataset)
            acc_num = at_once_predict(test_tokenized_inputs.input_ids,predict_id_list,label_id_list,tokenizer,output_dir)
            print(f"acc: {acc_num/len(test_labels)}")
            
        
        elif architecture_name == "iterative":
            test_inputs, test_labels = read_jsonl_files(test_dataset_names)            
            MAX_STEPS = 5
            acc_num = 0            
            data_size = len(test_labels)
            
            for i in range(MAX_STEPS):
                test_tokenized_inputs = tokenizer(test_inputs, padding=True, truncation=True, return_tensors='pt')
                test_tokenized_labels = tokenizer(test_labels, padding=True, truncation=True, return_tensors='pt')
                
                test_dataset = SimpleDataset(test_tokenized_inputs, test_tokenized_labels["input_ids"])
                predict_id_list,label_id_list, metrics = trainer.predict(test_dataset=test_dataset)
                
                each_acc_num, calculated_list, predict_text_list = iterative_predict(test_tokenized_inputs.input_ids,predict_id_list,label_id_list,tokenizer,output_dir,i)
                acc_num += each_acc_num
                
                predict_text_list = clean_predict_text(predict_text_list)
                test_inputs = list(map(lambda x,y: x + " , " + y ,test_inputs,predict_text_list))
                test_inputs, test_labels = eliminate_calculated_index(calculated_list,test_inputs, test_labels)
                if len(test_inputs) == 0 : break
                
            print(f"acc: {acc_num/data_size}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--architecture_name', default="sample")
    parser.add_argument('--train_dataset_names', default="train1,train2")
    parser.add_argument('--valid_dataset_names', default="valid1,valid2")
    parser.add_argument('--test_dataset_names', default="test1,valid2")
    parser.add_argument('--model_name', default="bart")
    parser.add_argument('--load_model_dir', default="facebook/bart-base") #fine-tuned model('./saved/checkpoint-480000')
    parser.add_argument('--train', default='false')
    parser.add_argument('--predict', default='false')
    parser.add_argument('--train_epochs', default=100, type=int)
    parser.add_argument('--eval_steps', default=100, type=int)
    parser.add_argument('--output_dir', default="save/test")
    parser.add_argument('--run_dir', default="save/test")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    args = parser.parse_args()
    main(args)
