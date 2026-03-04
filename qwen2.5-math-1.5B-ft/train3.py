import re
import os
import torch
import pandas as pd
from datasets import Dataset

# 1. 优化器改为从 torch.optim 导入
from torch.optim import AdamW

# 2. 调度器仍然从 transformers 导入
from transformers import get_scheduler

# 其它常用类从 transformers 顶层导入
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from peft import LoraConfig, get_peft_model, TaskType
import swanlab
# 更新 SwanLabCallback 导入路径
from swanlab.integration.transformers import SwanLabCallback

# === 路径配置（请根据实际情况修改） ===
MODEL_PATH = "/home/hzw/.cache/modelscope/hub/models/Qwen/Qwen2.5-math-7B/"
TRAIN_PATH = "math_train.jsonl"
TEST_PATH = "math_test.jsonl"

# === 加载 Tokenizer 和 模型（fp16） ===
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
    trust_remote_code=True
)
# 不使用 device_map，手动多卡
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
)
model.enable_input_require_grads()  # LoRA 需要开启梯度检查点支持

# === 数据预处理函数 ===
def process_func(example):
    MAX_LENGTH = 1024
    instruction = "你是一个代数老师，擅长解答各种代数问题，请根据问题一步一步推导认真作答。"
    prompt = (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    ins = tokenizer(prompt, add_special_tokens=False)
    res = tokenizer(example["output"], add_special_tokens=False)

    input_ids = ins["input_ids"] + res["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = ins["attention_mask"] + res["attention_mask"] + [1]
    labels = [-100] * len(ins["input_ids"]) + res["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# === 读取并构造 Dataset ===
train_df = pd.read_json(TRAIN_PATH, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# === LoRA 配置 ===
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    inference_mode=False,
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

# === 包装多卡 ===
if torch.cuda.is_available():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        print(f"→ 使用 {torch.cuda.device_count()} 张 GPU 进行并行训练")
        model = torch.nn.DataParallel(model)
else:
    device = torch.device("cpu")

model = model.to(device)

# === 分层学习率 ===
layerwise_params = {}
for name, param in model.named_parameters():
    if not ("lora_A" in name or "lora_B" in name):
        continue
    m = re.search(r"transformer\.h\.(\d+)\.", name)
    layer_id = int(m.group(1)) if m else 100
    layerwise_params.setdefault(layer_id, []).append(param)

base_lr = 1e-4
max_layer = max(layerwise_params.keys())
print(max_layer)
optimizer_grouped_parameters = []
for lid, params in layerwise_params.items():
    lr = base_lr * (0.1 + 0.9 * (1 - lid / float(max_layer)))
    optimizer_grouped_parameters.append({"params": params, "lr": lr})

optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8)

# === TrainingArguments ===
args = TrainingArguments(
    output_dir="./output/qwen2.5_math_7B_ft_1",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    save_strategy="epoch",
    save_total_limit=1,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    logging_steps=1,
    logging_dir="./logs",
    remove_unused_columns=False,  # 保留所有输入列以避免 DataParallel Wrapper 签名问题
)

# Scheduler
steps_per_epoch = len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
total_steps = steps_per_epoch * args.num_train_epochs
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# SwanLab Callback
swanlab_cb = SwanLabCallback(project="qwen2.5-math", experiment_name="qwen2.5-math-7B-ft-1")

# === Trainer ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    optimizers=(optimizer, scheduler),
    callbacks=[swanlab_cb],
)
trainer.train()

# === 推理函数（可选） ===
def predict(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_new_tokens=1024)
    gen_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

