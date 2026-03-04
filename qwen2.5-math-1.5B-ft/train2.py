import re
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
from swanlab.integration.huggingface import SwanLabCallback

# === 路径配置（请根据实际情况修改） ===
MODEL_PATH = "/home/hzw/.cache/modelscope/hub/models/Qwen/Qwen2.5-math-7B/"
TRAIN_PATH = "math_train.jsonl"
TEST_PATH = "math_test.jsonl"

# === 加载 Tokenizer 和 模型（fp16、自动 Device Map） ===
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
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
    instruction_tokenized = tokenizer(prompt, add_special_tokens=False)
    response_tokenized = tokenizer(example["output"], add_special_tokens=False)

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

# === 读取并构造 Dataset ===
train_df = pd.read_json(TRAIN_PATH, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# === LoRA 配置（全局秩 r=4） ===
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

# === 分层学习率：为不同层的 LoRA 参数分组赋予不同 lr ===
layerwise_params = {}
for name, param in model.named_parameters():
    # 只对 LoRA Adapter 的 lora_A / lora_B 矩阵做优化，原始权重已经被冻结
    if not (("lora_A" in name) or ("lora_B" in name)):
        continue

    # 提取参数所属的 Transformer 层编号，例如 "transformer.h.5.attention.q_proj.lora_A"
    m = re.search(r"transformer\.h\.(\d+)\.", name)
    if m:
        layer_id = int(m.group(1))
    else:
        # 如果不是 standard transformer block（比如 lm_head），归为较大编号
        layer_id = 100

    layerwise_params.setdefault(layer_id, []).append(param)

# 基准学习率与最大层编号
base_lr = 1e-4
max_layer_id = max(layerwise_params.keys())
optimizer_grouped_parameters = []
for layer_id, params in layerwise_params.items():
    # 线性插值生成学习率：layer_id=0 → lr = base_lr；layer_id=max_layer_id → lr = base_lr*0.1
    lr_for_this = base_lr * (0.1 + 0.9 * (1.0 - layer_id / float(max_layer_id)))
    optimizer_grouped_parameters.append({
        "params": params,
        "lr": lr_for_this
    })

# 使用 torch.optim.AdamW 并传入多个参数组（不同 lr）
optimizer = AdamW(
    optimizer_grouped_parameters,
    betas=(0.9, 0.999),
    eps=1e-8
)

# === TrainingArguments 配置 ===
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
    # 注意：这里不要再写 learning_rate，否则会被忽略
)

# 计算总训练步数
num_steps_per_epoch = len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
num_training_steps = num_steps_per_epoch * args.num_train_epochs

# 构造线性衰减 Scheduler（无 warmup）
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# === SwanLab 回调 ===
swanlab_callback = SwanLabCallback(project="qwen2.5-math", experiment_name="qwen2.5-math-7B-ft-1")

# === 构造 Trainer 并开始训练 ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    optimizers=(optimizer, scheduler),
    callbacks=[swanlab_callback],
)
trainer.train()

# === 推理部分（可选） ===
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

# 如果需要将测试集推理结果上报到 SwanLab，可按需保留下面这部分代码
test_df = pd.read_json(TEST_PATH, lines=True)
test_text_list = []
for _, row in test_df.iterrows():
    messages = [
        {"role": "system", "content": "你是一个代数老师，擅长解答各种代数问题，请根据问题一步一步推导认真作答。"},
        {"role": "user", "content": row['input']}
    ]
    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": response})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))

swanlab.log({"Prediction": test_text_list})
swanlab.finish()

