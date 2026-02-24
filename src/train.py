"""
URL Page Type Classifier Training
使用Qwen2.5-1.5B + LoRA微调
"""

import os
import json
import random
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# 配置
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
OUTPUT_DIR = "output"
DATA_PATH = "data/data.json"
SEED = 42

def prepare_data():
    """准备训练数据"""
    print("Preparing data...")
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    def format_prompt(sample):
        text = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{sample['instruction']}

URL: {sample['input']}<|im_end|>
<|im_start|>assistant
{sample['output']}<|im_end|>"""
        return {"text": text}
    
    formatted_data = [format_prompt(d) for d in data]
    return formatted_data

def tokenize_function(examples, tokenizer):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    random.seed(SEED)
    
    # 加载数据
    formatted_data = prepare_data()
    train_dataset = Dataset.from_list(formatted_data)
    
    # 加载模型
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
    )
    
    # 训练
    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    print("Starting training...")
    trainer.train()
    
    # 保存
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    
    print("Training complete!")

if __name__ == '__main__':
    main()
