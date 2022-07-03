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
from dentaku_tokenizer.tokenizer import BartDentakuTokenizer, T5DentakuTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import os
from typing import List, Dict
from datasets2strlist import sample_datasets2strlist,proofwriter_datasets2strlist_at_once,proofwriter_datasets2strlist_iterative,proofwriter_datasets2strlist_ansonly

from utils import retrieve_last_string,proofwriter_at_once_predict,proofwriter_iterative_predict,clean_predict_text,eliminate_calculated_index

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
    datasets_name = args.datasets_name
    train_dataset_name = args.train_dataset_name
    valid_dataset_name = args.valid_dataset_name
    test_dataset_name = args.test_dataset_name
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
    

    
    assert datasets_name in ["sample","proofwriter_at_once","proofwriter_iterative","proofwriter_ansonly"], "datasets_nameが間違っています"
    assert model_name in ["bart","reasoning_bart","t5"],"model_nameが間違っています"
    
    if model_name in ["bart","reasoning_bart"]:
        tokenizer = BartDentakuTokenizer.from_pretrained("facebook/bart-base")
    elif model_name in ["t5"]:
        #tokenizer = T5DentakuTokenizer.from_pretrained("t5-base")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
     
    # inputs,labels：List[str]    
    if datasets_name == "sample":
        train_inputs, train_labels = sample_datasets2strlist()
        valid_inputs, valid_labels = sample_datasets2strlist()
    elif datasets_name == "proofwriter_at_once":
        train_inputs, train_labels = proofwriter_datasets2strlist_at_once(train_dataset_name)
        valid_inputs, valid_labels = proofwriter_datasets2strlist_at_once(valid_dataset_name)
    elif datasets_name == "proofwriter_iterative":
        train_inputs, train_labels = proofwriter_datasets2strlist_iterative(train_dataset_name)
        valid_inputs, valid_labels = proofwriter_datasets2strlist_iterative(valid_dataset_name)
    elif datasets_name == "proofwriter_ansonly":
        train_inputs, train_labels = proofwriter_datasets2strlist_ansonly(train_dataset_name)
        valid_inputs, valid_labels = proofwriter_datasets2strlist_ansonly(valid_dataset_name)
                
    train_tokenized_inputs = tokenizer(
        train_inputs, padding=True, truncation=True, return_tensors='pt')
    train_tokenized_labels = tokenizer(
        train_labels, padding=True, truncation=True, return_tensors='pt')
    valid_tokenized_inputs = tokenizer(
        valid_inputs, padding=True, truncation=True, return_tensors='pt')
    valid_tokenized_labels = tokenizer(
        valid_labels, padding=True, truncation=True, return_tensors='pt')
    
    
    #ids = tokenizer("text","text")["input_ids"]
    #print(ids)
    #print([tokenizer.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False) for x in ids])
    #print([tokenizer.decode([0,1,2], skip_special_tokens=False, clean_up_tokenization_spaces=False)])
    #print([tokenizer.decode(train_tokenized_inputs.input_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)])
    #exit()
    
 
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
        if datasets_name in ["proofwriter_ansonly","proofwriter_at_once"]:
            if datasets_name == "proofwriter_ansonly":
                test_inputs, test_labels  = proofwriter_datasets2strlist_ansonly(test_dataset_name)
            else:
                test_inputs, test_labels  = proofwriter_datasets2strlist_at_once(test_dataset_name)
            test_tokenized_inputs = tokenizer(test_inputs, padding=True, truncation=True, return_tensors='pt')
            test_tokenized_labels = tokenizer(test_labels, padding=True, truncation=True, return_tensors='pt')
            test_dataset = SimpleDataset(test_tokenized_inputs, test_tokenized_labels["input_ids"])
            predict_id_list,label_id_list, metrics = trainer.predict(test_dataset=test_dataset)
            acc_num = proofwriter_at_once_predict(test_tokenized_inputs.input_ids,predict_id_list,label_id_list,tokenizer,output_dir)
            print(f"acc: {acc_num/len(test_labels)}")
            
        
        elif datasets_name == "proofwriter_iterative":
            MAX_STEPS = 5
            acc_num = 0            
            test_theories, test_questions, test_labels = proofwriter_datasets2strlist_iterative(test_dataset_name,test=True)
            data_size = len(test_labels)
            
            for i in range(MAX_STEPS):
                test_inputs = list(map(lambda x,y: x + " </s> " + y ,test_theories,test_questions))
                test_tokenized_inputs = tokenizer(test_inputs, padding=True, truncation=True, return_tensors='pt')
                test_tokenized_labels = tokenizer(test_labels, padding=True, truncation=True, return_tensors='pt')
                
                test_dataset = SimpleDataset(test_tokenized_inputs, test_tokenized_labels["input_ids"])
                predict_id_list,label_id_list, metrics = trainer.predict(test_dataset=test_dataset)
                
                each_acc_num, calculated_list, predict_text_list = proofwriter_iterative_predict(test_tokenized_inputs.input_ids,predict_id_list,label_id_list,tokenizer,output_dir,i)
                acc_num += each_acc_num
                
                predict_text_list = clean_predict_text(predict_text_list)
                test_theories = list(map(lambda x,y: x + " " + y ,test_theories,predict_text_list))
                test_theories, test_questions, test_labels = eliminate_calculated_index(calculated_list,test_theories, test_questions, test_labels)
                if len(test_theories) == 0 : break
                
            print(f"acc: {acc_num/data_size}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--datasets_name', default="sample")
    parser.add_argument('--train_dataset_name', default="train")
    parser.add_argument('--valid_dataset_name', default="valid")
    parser.add_argument('--test_dataset_name', default="test")
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
