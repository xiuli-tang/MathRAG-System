import json
import pandas as pd
import torch
from datasets import Dataset
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer
import os
import swanlab

# === 模型路径 ===（请改为你本地模型的路径）
MODEL_PATH = "/home/hzw/.cache/modelscope/hub/models/Qwen/Qwen2.5-math-1.5B/"
TRAIN_PATH = "math_train.jsonl"
TEST_PATH = "math_test.jsonl"

# === 加载模型 & tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
model.enable_input_require_grads()  # LoRA使用梯度检查点需要

# === 数据预处理函数 ===
def process_func(example):
    MAX_LENGTH = 1024
    instruction = "你是一个代数老师，擅长解答各种代数问题，请根据问题认真作答。"
    prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction_tokenized = tokenizer(prompt, add_special_tokens=False)
    response_tokenized = tokenizer(example['output'], add_special_tokens=False)

    input_ids = instruction_tokenized["input_ids"] + response_tokenized["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction_tokenized["attention_mask"] + response_tokenized["attention_mask"] + [1]
    labels = [-100] * len(instruction_tokenized["input_ids"]) + response_tokenized["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# === 加载数据集 ===
train_df = pd.read_json(TRAIN_PATH, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# === LoRA 配置 ===
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, config)

# === 训练参数 ===
args = TrainingArguments(
    output_dir="./output/qwen2.5_math_1.5B_ft_1",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,  # 每步都输出训练日志
    num_train_epochs=2,  # 原来是2，改成10轮
    save_strategy="epoch",  # 每个 epoch 保存一次模型
    save_total_limit=1,  # 最多保存两个 checkpoint，避免磁盘爆满
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",  # 同时输出到 swanlab + 控制台
    logging_dir="./logs",  # 保存 TensorBoard 日志
)

# === SwanLab callback ===
swanlab_callback = SwanLabCallback(project="qwen2.5-math", experiment_name="qwen2.5-math-1.5B-ft-1")

# === 启动训练 ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)
trainer.train()

# === 推理函数 ===
def predict(messages, model, tokenizer):
    device = "cuda:0"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response

# === 用测试集做前向预测并记录到 SwanLab ===
test_df = pd.read_json(TEST_PATH, lines=True)
test_text_list = []

for index, row in test_df.iterrows():
    messages = [
        {"role": "system", "content": "你是一个代数老师，擅长解答各种代数问题，请根据问题认真作答。"},
        {"role": "user", "content": row['input']}
    ]
    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": response})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))

swanlab.log({"Prediction": test_text_list})
swanlab.finish()
