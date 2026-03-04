from flask import Flask, request, render_template_string, jsonify, render_template, url_for, send_from_directory, \
    send_file
from flask_cors import CORS
import os
import re
import logging
from datetime import datetime
import requests
from concel import replace_single_dollar
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda:1"  # 设置设备为 CUDA
# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "/home/hzw/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct",  # 使用正确的路径
    torch_dtype=torch.float16,  # 自动选择数据类型
    device_map=device  # 自动映射到可用设备
)

tokenizer = AutoTokenizer.from_pretrained("/home/hzw/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct")


def get_q(prompt):
    messages = [
        {"role": "system",
         "content": "你是一个智能数学教学助手，请根据以下历史对话内容，自动识别使用的语言，从中识别用户感兴趣的数学主题或问题，并设计5个循序渐"
                    "进的问题，帮助用户更深入地理解和掌握该主题。"
                    "回答请严格按照以下格式为：\n 1. **问题1**\n2. **问题2**\n3. **问题3**\n4. **问题4**\n5. **问题5**\n"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(res)
    return res


log = logging.getLogger('werkzeug')
log.disabled = False
app = Flask(__name__)
app.logger.disabled = False
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/home/hzw/code/flask')
app.template_folder = TEMPLATE_DIR
CORS(app)  # 允许所有域访问
UPLOAD_FOLDER = os.getcwd()  # 当前目录
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 允许的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return '没有文件部分'
    file = request.files['file']

    # 如果文件类型合法
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], f"1.{str(file.filename).split('.')[1]}")
        # 保存文件到当前目录
        file.save(filename)
        # out = p_to_t(filename)
        out ="求函数的导数： $$ y=\operatorname{l n} ( 1+\mathrm{e}^{x} )-x. $$"
        print(out)
        return jsonify({"bold_text":out})  # 将公式传递给模板

    return '不支持的文件格式'



import os


def list_txt_files(directory):
    txt_files = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        return txt_files
    except FileNotFoundError:
        print(f"错误：指定的目录 {directory} 未找到。")
    except PermissionError:
        print(f"错误：没有权限访问目录 {directory}。")
    except Exception as e:
        print(f"发生未知错误：{e}")
    return []


@app.route('/rel', methods=['get', 'post'])
def rel():
    hist = str(request.get_json()['question'])
    # questions = re.findall(r'(\.)(.*?)', get_q(hist))
    # # 拼接编号和问题
    # question_list = [f"{num} {q}" for num, q in questions]
    question_list = get_q(hist).split('\n')
    print(question_list)
    return jsonify({"bold_text": [q.replace('*', '') for q in question_list], "message": "None"})


@app.route('/pb', methods=['GET'])
def get_data():
    id = request.args.get('id', type=int)
    is_dark = request.args.get('is_dark', type=int)
    data = [
        r"""
        直线的点斜式方程：

        $$y - y_1 = k(x - x_1)$$

        其中 $(x_1, y_1)$ 是直线上一点，$k$ 是直线的斜率。
        """,
        r"""
        直线的斜截式方程：

        $$y = kx + b$$

        其中 $k$ 是直线的斜率，$b$ 是直线在 $y$ 轴上的截距。
        """,
        r"""
        直线的两点式方程：

        $$\frac{y - y_1}{y_2 - y_1} = \frac{x - x_1}{x_2 - x_1}$$

        其中 $(x_1, y_1)$ 和 $(x_2, y_2)$ 是直线上两点。
        """,
        r"""
        直线的一般式方程：

        $$Ax + By + C = 0$$

        其中 $A, B, C$ 为常数，且 $A$ 和 $B$ 不同时为零。
        """,
        r"""
        直线的截距式方程（当 $A \neq 0$ 且 $B \neq 0$ 时）：

        $$\frac{x}{a} + \frac{y}{b} = 1$$

        其中 $a$ 和 $b$ 分别是直线在 $x$ 轴和 $y$ 轴上的截距。
        """
    ]

    data = ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"]
    for i in range(len(data)):
        data[i] = replace_single_dollar(data[i])
    if is_dark == 1:
        return render_template('show_dark.html', id=id)
    else:
        return render_template('show.html', id=id)


@app.route('/background-image')
def get_background_image():
    return send_file('team-logo2.png', mimetype='image/png')


@app.route('/get_files')
def get_files():
    id = int(request.args.get('id'))
    li = ['导数的基本运算与几何意义.txt', '函数的单调性与导数.txt', '函数的极值与最值.txt', '导数的应用：不等式证明.txt',
          '导数的应用：零点问题.txt', '含参问题的讨论.txt']
    print(li)
    return jsonify(li)


@app.route('/get_file_content')
def get_file_content():
    filename = request.args.get('filename')
    with open(f"{TEMPLATE_DIR}/db/{filename}", 'r', encoding='utf-8') as f:
        content = f.read()
    return jsonify(content.split("------------------------------------------------------------"))


@app.route('/get_all_files')
def get_all_files():
    files_info = []
    for root, dirs, files in os.walk(os.path.join(TEMPLATE_DIR, 'db')):
        for file in files:
            if file.endswith('.txt'):  # 只选择.txt文件
                file_path = os.path.join(root, file)

                # 获取文件大小
                size = os.path.getsize(file_path)

                # 获取文件的最后修改时间
                modified_timestamp = os.path.getmtime(file_path)
                modified_time = datetime.utcfromtimestamp(modified_timestamp).strftime('%Y-%m-%dT%H:%M:%SZ')

                # 构建文件信息字典
                file_info = {
                    "name": file,
                    "size": size,
                    "modified": modified_time,
                    "type": "document"
                }

                # 添加到文件列表中
                files_info.append(file_info)

    return jsonify(files_info)


# 自定义 IP 和端口
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
