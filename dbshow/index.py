from flask import Flask, jsonify, request, send_file
import os
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许所有域访问
# 配置您的文件系统根目录
BASE_DIR = r"/home/hzw/code/dbshow/knowledge_db"

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/get_directory')
def get_directory():
    path = request.args.get('path', '')
    full_path = os.path.join(BASE_DIR, path)
    
    # 安全校验
    if not os.path.exists(full_path) or not full_path.startswith(BASE_DIR):
        return jsonify({"error": "Invalid path"}), 400
    
    try:
        items = os.listdir(full_path)
        directories = []
        files = []
        
        for item in items:
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                directories.append(item)
            elif os.path.isfile(item_path):
                files.append(item)
        
        return jsonify({
            "path": path,
            "directories": sorted(directories),
            "files": sorted(files)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/background-image')
def get_background_image():
    return send_file(r'F:\python_project\big_model\flask\team-logo2.png', mimetype='image/png')
@app.route('/get_file_content')
def get_file_content():
    file_path = request.args.get('file_path', '')
    full_path = os.path.join(BASE_DIR, file_path)
    
    # 安全校验
    if not os.path.exists(full_path) or not full_path.startswith(BASE_DIR):
        return jsonify({"error": "Invalid file path"}), 400
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式按分隔符分割内容
        # 分隔符是至少6个连续的破折号
        sections = re.split(r'-{6,}', content)
        
        # 过滤空段落并添加HTML格式
        processed = []
        for section in sections:
            stripped = section.strip()
            if stripped:
                # 保留原始换行符，替换为HTML换行
                html_content = stripped.replace('\n', '<br>')
                processed.append(html_content)
        
        return jsonify({
            "file_path": file_path,
            "content": processed
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5004, debug=False)