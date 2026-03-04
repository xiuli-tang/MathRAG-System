from flask import Flask, request, render_template_string, jsonify, render_template, url_for, send_from_directory
import os
from flask_cors  import CORS
# from p_to_t import p_to_t


app = Flask(__name__)
UPLOAD_FOLDER = os.getcwd()  # 当前目录
CORS(app)  # 允许所有域访问
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
        out = "解方程：5 $( \mathrm{x} \ +\ 2 ) \ =\ 3 0$ "
        print(out)
        return jsonify({"bold_text":out})  # 将公式传递给模板

    return '不支持的文件格式'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
