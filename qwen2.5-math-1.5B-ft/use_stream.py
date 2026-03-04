import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


from transformers import TextIteratorStreamer
import threading

def stream_predict(messages, model, tokenizer):
    device = "cuda"

    # 构造 prompt
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

    # 初始化 streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 设置线程生成
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 实时输出每个 token
    response = ""
    for new_text in streamer:
        print(new_text, end='', flush=True)
        response += new_text

    return response


# 模型路径
# model_dir = "F:\\python_project\\big_model\\Qwen2.5-Math-1.5B-Instruct"
model_dir = "/home/hzw/.cache/modelscope/hub/models/Qwen/Qwen2.5-math-7B/"
# lora_dir = "F:\\python_project\\big_model\\qwen_math_ft\\output\\qwen2.5_math_ft-2\\checkpoint-60"
lora_dir = "/home/hzw/code/qwen2.5-math-1.5B-ft/output/qwen2.5_math_7B_ft_1/checkpoint-35"
# 加载原模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)

# 加载LoRA模型
model = PeftModel.from_pretrained(model, model_id=lora_dir)

input_text = r"""
解二次方程 \(x^2 + 4x + 3 = 0\)
"""
test_texts = {
    "instruction": """
                你是一个代数老师，擅长解答各种代数问题，请根据问题一步一步推导认真作答。
                """,
    "input": f"{input_text}"
}

instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]

# 流式生成预测
response = stream_predict(messages, model, tokenizer)
print("\nFinal Response:", response)

