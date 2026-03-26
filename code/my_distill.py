#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CMD: cd Downloads/llm_related/knowledge_distillation_llm
#      conda activate train_env
#      python my_distill.py 

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding_id = tokenizer.pad_token_id
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        line = self.data[index]
        # instruction_text = line['instruction']
        # input_text = line['input']
        # query = instruction_text + input_text
        query = line['query']
        output_text = line['response']
        answer = output_text + self.tokenizer.eos_token
        messages = []
        messages.append({'role': 'user', 'content': query})   
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
        
        prompt_input_ids = self.tokenizer.encode(prompt)
        answer_input_ids = self.tokenizer.encode(answer)
        
        input_ids = prompt_input_ids + answer_input_ids
        labels = [-100] * len(prompt_input_ids) + answer_input_ids
        attention_mask = [1] * len(input_ids)
        text_len = len(input_ids)
        
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]
        else:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - text_len)
            labels = labels + [-100] * (self.max_seq_len - text_len)
            attention_mask = attention_mask + [0] * (self.max_seq_len - text_len)
        
        # input_ids = input_ids[:-1]
        # labels = labels[1:]
        return {'input_ids': torch.tensor(input_ids), 'attention_mask':torch.tensor(attention_mask), 'labels': torch.tensor(labels)}
    


# In[2]:


# 计算前向kl散度
def compute_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="sum",
        temp = 1.0, 
        
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp

        log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kl = (teacher_probs * (teacher_log_probs - log_probs)) 
        kl = kl.sum(-1)
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        if reduction == "sum":
            
            kl = kl.sum(dim=1)
        
        elif reduction == "mean":
            kl = kl.sum(dim=1) / (~pad_mask).sum(dim=1)

        return kl


# In[3]:


from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from swanlab.integration.transformers import SwanLabCallback

# from dataset import SFTDataset
# from utils import compute_fkl, compute_rkl, compute_skewed_fkl, compute_skewed_rkl



class KGTrainer(Trainer):
    
    def __init__(
        self,
        model = None,
        teacher_model = None,
        if_use_entropy = False,
        args = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        model_init = None, 
        compute_metrics = None, 
        callbacks = None,
        optimizers = (None, None), 
        preprocess_logits_for_metrics = None,
        # tokenizer=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            # tokenizer=tokenizer
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        loss = outputs.loss
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # 如果教师模型和学生模型输出形状不匹配，对学生模型进行padding或对教师模型进行截断
        if logits.shape[-1] != teacher_logits.shape[-1]:
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        
        labels = inputs['labels']
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=1.0).mean()
        
        if self.if_use_entropy:
            alpha = 0.01
            loss_total = alpha * kl + (1-alpha) * loss
            loss_total = alpha * kl + loss
        else:
            loss_total = loss
        
        return (loss_total, outputs) if return_outputs else loss_total


class BasicTrainer(Trainer):
    
    def __init__(
        self,
        model = None,
        # teacher_model = None,
        # if_use_entropy = False,
        args = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        model_init = None, 
        compute_metrics = None, 
        callbacks = None,
        optimizers = (None, None), 
        preprocess_logits_for_metrics = None,
        # tokenizer=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            # tokenizer=tokenizer
        )
        # self.teacher_model = teacher_model
        # self.if_use_entropy = if_use_entropy
        
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        outputs = model(**inputs)
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss_total = loss
        
        return (loss_total, outputs) if return_outputs else loss_total

# In[4]:


if __name__ == '__main__':
    
    # 学生模型
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    lora_config = LoraConfig(
        r=4,  
        lora_alpha=8,  
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, 
        task_type=TaskType.CAUSAL_LM
    )
    
    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    model.cuda()
    print(model.print_trainable_parameters())
    print(f'Load Student Model Complete.\n')
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    # 教师模型，在给定数据上通过lora微调
    # teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")

    # teacher_model.cuda()
    # teacher_model.eval()

    # print(f'Load Teacher Model Complete.\n')

    args = TrainingArguments(output_dir='./results_0223_exp5', 
                            num_train_epochs=5, 
                            do_train=True, 
                            per_device_train_batch_size=2,
                            per_device_eval_batch_size=2,
                            gradient_accumulation_steps=16,
                            logging_steps=100,             # 每 100 步记录一次训练损失
                            eval_strategy="steps",         # 将 epoch 改为 steps
                            eval_steps=100,                # 每 100 步进行一次验证
                            report_to='none',
                            save_strategy='steps',
                            save_steps=100,
                            save_total_limit=10,
                            bf16=True,
                            learning_rate=0.00006,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
    )

    data_collator = DefaultDataCollator()
    train_dataset = SFTDataset('data/data_train_large_math10000.json', tokenizer=tokenizer, max_seq_len=1024)
    val_dataset = SFTDataset('data/data_valid_large_math10000.json', tokenizer=tokenizer, max_seq_len=1024)

    swanlab_callback = SwanLabCallback(
        project="qwen2.5-3b-distill",
        workspace="mtm22",
        experiment_name="SFT-0302-exp6" ,
        # experiment_id="cdhfh6fwq6x5qxftusmc6"
    )

    # trainer = KGTrainer(model=model,
    #                     teacher_model=teacher_model, 
    #                     if_use_entropy = False,
    #                     args=args, 
    #                     train_dataset=train_dataset, 
    #                     eval_dataset=val_dataset,
    #                     callbacks=[swanlab_callback],
    #                     # tokenizer=tokenizer, 
    #                     data_collator=data_collator)
    
    trainer = BasicTrainer(model=model,
                        args=args, 
                        train_dataset=train_dataset, 
                        eval_dataset=val_dataset,
                        callbacks=[swanlab_callback],
                        # tokenizer=tokenizer, 
                        data_collator=data_collator)

    # In[ ]:


    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves_0302_exp6')
    trainer.save_state()



# In[ ]:




