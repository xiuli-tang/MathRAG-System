import json
import base64
import requests
import re
import time
import sys
sys.stdout.reconfigure(line_buffering=True)


# Base64 编解码函数
def chinese_to_base64(text):
    return base64.b64encode(text.encode('utf-8')).decode('ascii')

def base64_to_chinese(encoded_text):
    return base64.b64decode(encoded_text.encode('ascii')).decode('utf-8')

# 提取 JSON
def extract_json_from_output(output: str):
    try:
        match = re.search(r"\[\s*{.*?}\s*\]", output, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            print("❗未找到 JSON 数组")
            return []
    except json.JSONDecodeError as e:
        print("❌ JSON 解码失败:", e)
        return []

# API 调用
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxx"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

def call_deepseek(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    response = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# 主逻辑
def process_json_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        datas = json.load(f)

    results = []
    batch = []

    for idx, item in enumerate(datas):
        item_id = item["id"]
        decoded_content = base64_to_chinese(item["content"])
        batch.append({"id": item_id, "content": decoded_content})

        if len(batch) == 5 or idx == len(datas) - 1:
            # 构造 prompt
            contents_str = "\n".join([f"ID: {x['id']}\n{x['content']}" for x in batch])
            prompt = (
                "以下是10道题目和解析的混合内容，每题格式混杂，请你将它们拆分为 JSON 数组，"
                "每项为 {\"id\": \"原id\", \"question\": \"题目部分\", \"answer\": \"解析部分\"}。"
                "注意：不要生成多余内容，保持 JSON 格式严格正确。\n\n"
                f"{contents_str}"
            )

            print(f"🔄 正在处理第 {idx+1} 条内容（共 {len(datas)} 条）...")

            if idx+1 not in [30,35,40,45,50,55,60,65,70,75,95,90,100,105,180,185,170,175,190,195]:
                print(f'{idx+1}跳过')
                batch=[]
                continue
            try:
                response_text = call_deepseek(prompt)
                print(response_text)
                parsed = extract_json_from_output(response_text)
                if len(parsed) == 0:
                    print(">>>>>>>>>>> idx <<<<<<<<<<<<")
                print(parsed)
                # Base64 编码处理
                for item in parsed:
                    item["question"] = chinese_to_base64(item["question"])
                    item["answer"] = chinese_to_base64(item["answer"])
                results.extend(parsed)

            except Exception as e:
                print(f"❌ 错误发生于 batch {idx // 10 + 1}: {e}")
            finally:
                batch = []  # 清空

            time.sleep(2)  # 防止请求过快

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ 全部处理完成，结果保存至 {output_file}")

# 调用主函数
if __name__ == "__main__":
    process_json_file("300ticon.json", "s2plit_encoded_result.json")
