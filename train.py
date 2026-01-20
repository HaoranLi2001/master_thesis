# pip install torch transformers peft accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraModel, LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from transformers import Trainer, TrainingArguments
from utils import forward_KLD
from dataset import SFTDataset

def log_memory(step_name):
    # 获取当前时间
    now = datetime.datetime.now().strftime('%H:%M:%S')
    # 已分配显存 (正在使用的)
    allocated = torch.cuda.memory_allocated() / 1024**2
    # 预留显存 (PyTorch 占用的总坑位)
    reserved = torch.cuda.memory_reserved() / 1024**2
    # 显存峰值
    max_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    log_str = f"[{now}] {step_name} -> 已分配: {allocated:.2f}MB, 预留: {reserved:.2f}MB, 峰值: {max_mem:.2f}MB\n"
    
    # 实时写入文件
    with open("gpu_memory_log.txt", "a") as f:
        f.write(log_str)
    print(log_str) # 同时在终端打印

class KD_Trainer(Trainer):
    def __init__(self, model, teacher_model, if_use_entropy,*args,**kwargs):
        super().__init__(model=model, *args, **kwargs)
        # self.model = model
        self.teacher = teacher_model
        self.if_use_entropy = if_use_entropy

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)

        loss = outputs.loss
        logits_s = outputs.logits
        logits_t = teacher_outputs.logits

        # size:[batch, seq_length, vocab, logits_prob], 如果输出logits大小不匹配，截断教师模型
        if logits_s.shape[-1] != logits_t.shape[-1]:
            logits_t = logits_t[:, :, :logits_s.shape[-1]]

        labels = inputs['labels']

        kl = forward_KLD(logit_s=logits_s, logit_t= logits_t, targets= labels, padding_id=-100)
        if self.if_use_entropy == True:
            loss_total = kl * 0.5 + loss * 0.5
        else:
            loss_total = kl

        return (loss_total, outputs) if return_outputs else loss_total
    
if __name__ == "__main__":
    # Load model directly
    try:        
        teacher_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            dtype=torch.bfloat16, # 比 .half() 更稳
            device_map="auto"
        )
    except Exception as e:
        log_memory(f"!!! 发生崩溃 !!! 错误信息: {str(e)}")
        raise e

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        r=2,
        lora_alpha=64, 
        target_modules=target_modules, 
        lora_dropout=0.1, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    model.cuda() # 所有张量和模型都默认保存在CPU中，使用.cuda()转移到GPU，必须要在get_peft之后调用

    print(model.print_trainable_parameters())

    # 查看显存
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # lora_path = 
    args = TrainingArguments(
        output_dir = "./results",
        num_train_epochs=3, # epoch数量
        do_train=True, # 要训练，否则只验证和测试
        per_device_train_batch_size=1, # 每个gpu设备的“小”batch，即一次forward+backward计算多少个数据
        gradient_accumulation_steps=32, # 每N个“小”batch构成一个实际batch，即更新一次参数，实际batch = per_device_train_batch_size * gradient_accumulation_steps
                                       # 因此，要节省显存就要per_device_train_batch_size小，gradient_accumulation_steps大，即时间换空间
        # gradient_checkpointing=True, # 开启激活重计算，以节省显存
        logging_steps=10,
        save_total_limit=10,
        bf16=True,
        lr_scheduler_type="cosine",
        learning_rate=0.005,
        dataloader_num_workers=8
    )
    data_collator = DefaultDataCollator()
    dataset = SFTDataset("data.json", tokenizer=tokenizer, max_seq_len=512)

    log_memory(f"Load Dataset")

    trainer = KD_Trainer(
        model=model,
        teacher_model=teacher_model,
        if_use_entropy=True,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    log_memory(f"Set up trainer")
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model("./saves")
    trainer.save_state()