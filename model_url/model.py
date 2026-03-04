from flask_cors import CORS

from flask import Flask, request, Response, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from datetime import datetime
import logging

log = logging.getLogger('werkzeug')
log.disabled = True
app = Flask(__name__)
app.logger.disabled = True
CORS(app)
# 设备选择
device = "cuda"  if torch.cuda.is_available() else "cpu"

# 加载模型和 Tokenizer
model_name = "Qwen2.5-math-7B"
model_path = f"/home/hzw/.cache/modelscope/hub/models/Qwen/{model_name}"

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
def is_all_english_unicode(s):
    for char in s:
        if char.isspace():
            continue
        if not (65 <= ord(char.upper()) <= 90):
            return False
    return True

@app.route("/generate", methods=["POST"])
def generate():
    """流式生成大模型响应"""
    try:
        data = request.get_json()

        # if not prompt:
        #     return jsonify({"error": "No prompt provided"}), 400
        print(data)
        messages = data.get("messages")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        input_ids = model_inputs.input_ids
        max_new_tokens = 1024

        def stream_response():
            """流式返回 JSON"""
            generated_ids = input_ids.clone()
            for _ in range(max_new_tokens):
                outputs = model(input_ids=generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
                token_str = tokenizer.decode(next_token, skip_special_tokens=True).strip()

                if token_str:
                    if is_all_english_unicode(token_str) and len(token_str) > 1:
                        token_str = token_str+" "

                    response_data = {
                        "model": model_name,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "message": {"role": "assistant", "content": token_str},
                        "done": False
                    }
                    yield json.dumps(response_data, ensure_ascii=False) + "\n"

                if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                    break

            yield json.dumps({"done": True}) + "\n"

        return Response(stream_response(), content_type="application/json; charset=utf-8", headers={"Transfer-Encoding": "chunked"})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

