import torch
import json
from torch.utils.data import Dataset

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = tokenizer.pad_token_id
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data = self.data[index]

        # 应该是查看每列数据的标题
        instruction_text = data['instruction']
        input_text = data['input']
        output_text = data['output']

        query = instruction_text + input_text
        answer = output_text + self.tokenizer.eos_token

        messages = []
        messages.append({"role": "user", "content": query})

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        prompt_ids = self.tokenizer.encode(prompt)
        answer_ids = self.tokenizer.encode(answer)

        input_ids = prompt_ids + answer_ids
        input_len = len(input_ids)
        attention_mask = [1] * input_len
        labels = [-100] * len(prompt_ids) + answer_ids

        # 如果超出最大长度，截断
        if input_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]
        # 如果没有，填充到最大长度
        else: 
            input_ids = input_ids + [self.tokenizer.eos_token_ids] * (self.max_seq_len - input_len)
            labels = labels + [-100] * (self.max_seq_len - input_len) # -100不需要关注，其余非-100的token_id（回答部分，即要预测的部分）参与计算loss
            attention_mask = attention_mask + [0] * (self.max_seq_len - input_len) # 1的部分为有效上下文，0为填充padding
            
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}