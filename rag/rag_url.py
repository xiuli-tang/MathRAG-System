import os
import json
import base64
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import sys


app = Flask(__name__)
CORS(app)
device = "cuda:3"

# model_name = "Qwen2.5-Math-1.5B-Instruct"
model_name = "Qwen2___5-1___5B-Instruct"
# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    f"/home/hzw/.cache/modelscope/hub/models/Qwen/{model_name}",  # 使用正确的路径
    torch_dtype=torch.float16,  # 自动选择数据类型
    device_map=device,  # 自动映射到可用设备
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    f"/home/hzw/.cache/modelscope/hub/models/Qwen/{model_name}",
    local_files_only=True
)



# Please reason step by step, and put your final answer within \\boxed{}.
# Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}
@app.route("/rag", methods=["GET"])
def rag():
    prompt = request.args.get('prompt', type=str)
    messages = [
        {"role": "system", "content": """
        根据对话内容，分析用户的问题，从两个列表里选出最匹配的一个或多个关键词，严格只输出关键词
        列表一['分式', '方程', '直线方程', '导数', '集合与常用逻辑用语', '几何定理', '平面向量', '代数', '不等式', '三角函数', '数列', '函数', '一元二次函数，方程与不等式', '排列组合', '绝对值', '几何', '统计与概率', '解三角形'],
        列表二['正弦定理的证明', '圆的方程求解', '直线与圆的方程的应用', '位置关系判断', '余弦定理的证明', '斜率与倾斜角', '杨辉三角应用', '几何对称', '向量方程', '二项式定理应用', '微分中值定理（拉格朗日）', '正切函数', '互斥概率', '数学归纳法', '阶乘公式', '二项式定理例题', '不等式恒成立问题', '几何圆与直线', '累加法', '截距式方程例题', '倍角', '几何三角形', '绝对值数值求解', '前n项和性质', '斜率定义', '用不等式（组）表示不等关系', '函数的基本性质-最值', '等差数列例题', '函数数列', '综合边角关系问题', '几何垂直', '分式不等式解法', '贝叶斯公式应用', '函数单调性与导数的关系', '绝对值性质', '斜截式方程', '等差数列求通项公式', '斜率最值问题', '函数二次函数', '函数平移', '余弦定理与勾股定理的关系', '两圆位置关系', '等差数列的前n 项和', '几何菱形', '优先选择计算简单的定理', '二项展开式中系数和的求法', '含参不等式解法', '几何弦切角', '方程等式', '递增数列', '两直线交点计算', '方程形式转换', '中线方程求解', '平均分组有归属问题', '事件类型定义', '不等式比较方法', '三角函数的概念', '三角恒等变换', '等比数列的应用', '实际应用问题', '导数与函数极值', '函数极值', '直线方程选择', '斜率范围问题', '等差数列的判定方法', '代数整数', '绝对值不等式解法', '等差数列的判断', '点斜式方程应用', '平面向量运算的坐标表示', '绝对值取值范围', '限定分组有归属问题', '全称量词与存在量词', '等比数列前n项和公式——错位相减法', '非平均分组无归属问题', '事件独立性判定', '两直线垂直判定', '调和数列公式', '任意角和弧度制', '方程一元一次方程', '插空法', '实数大小比较的依据', '函数有界性', '向量的运算', '几何相似', '判断是否为等差数列项', '解三角形例题', '导数的概念', '中心对称性质', '累乘法', '距离肌酸', '多解问题', '二项分布', '点与圆的位置关系', '光线反射问题', '解三角形时忽略对角的讨论', '均值不等式', '全概率公式应用', '密码尝试概率', '频率与概率关系', '分式判别式', '恒成立/能成立/恰成立问题', '函数极限', '数列的综合应用', '几何角平分线', '正弦定理在解三角形中的应用', '排列性质', '函数最大最小值', '非平均分组有归属问题', '几何切点', '坐标法证明平面几何', '几何全等三角形', '平行直线系', '函数反比例函数', '对立事件', '向量的数量积', '截距与角度问题', '正、余弦定理的简单应用', '方程根与系数', '互斥事件加法公式', '测量距离问题的基本类型和解决方案', '空间坐标表示', '二项分布期望', '几何余角', '等比数列性质', '幂函数', '平行四边形判定', '几何等腰梯形', '代数倒数', '几何正方形', '最值定理口诀', '几何圆周角', '函数的基本性质-奇偶数', '求周期数列的前$n$ 项和问题', '倾斜角定义', '等差数列通项公式', '求等差数列前n项和', '斜率存在性', '平面向量的概念', '左右同正不等式', '代数方程关系', '利用斜率证明三点共线的方法', '平行线截线段问题', '余弦定理在解三角形中的应用', '代数合数', '方差性质', '绝对值零点', '非完全平均分组有归属问题', '分步乘法计数原理', '代数技巧', '对数函数', '方程去括号', '过定点直线系', '条件概率性质', '数列求和问题', '对边的比例关系的认识', '对数', '绝对值不等式', '几何圆柱', '距离最值问题', '常见不等式', '几何直线', '独立性定义', '求周期数列的项', '代数数位规则', '集合间的关系', '二次函数', '函数一次函数', '向量的数乘运算', '从 n 个元素中取 m 个元素的排列，常用方法', '平行方程', '分类加法计数原理', '等比数列通过公式推导——累乘法', '代数百分数', '几何切线', '信号传输概率', '几何长方体', '忽略构成三角形的条件致错', '特殊规定', '代数四则运算', '二项式的应用', '几何垂径', '圆的标准方程', '等比数列等差中项', '求通项公式', '等比数列前n项和', '代数自然数', '几何中位线', '全概率公式定义', '独立重复试验定义', '三点共线证明', '等式性质与不等式性质', '函数断点', '用不等式组表示不等关系', '两条直线位置关系的判定', '轨迹与圆位置关系', '集合的基本运算', '通项公式为S_n 与a_n的关系式', '正态分布', '分式列方程', '函数方程', '代数方程', '几何型概率性质', '条件概率定义', '必然事件概率', '等差数列通项公式性质', '函数定义', '绝对值不等式性质', '轨迹定义', '函数图像与性质', '几何割线', '几何圆', '绝对值方程', '几何直角三角形', '特殊数列', '不等式性质的应用——判断命题的真假', '几何平行线', '几何公共弦', '性质', '对余弦定理的理解', '不同函数增长的差异', '代数代数计算', '两组元素各相同的插空', '基本不等式的常用结论', '圆的切线方程', '斐波那契数列公式', '圆方程特点', '等差数列的前n 项和公式', '二项式系数', '函数连续', '二项式应用', '对立事件性质', '不等式性质的应用——证明不等式', '不等关系的实际应用', '余弦定理，正弦定理', '不等式', '二项展开式特点', '等比数列通项公式', '分式解法', '绝对值推论', '两点分布期望', '等差数列递推公式', '排列应用', '各二项式系数的和', '充分条件与必要条件', '诱导公式', '数列中的基本量', '排列数定义', '平面向量基本定理', '几何三角形边的性质', '利用直线方程判定直角三角形', '距离相等问题', '面积公式', '集合的概念', '通项公式', '点到直线距离', '二项式定理', '方程判别式', '分式化简', '独立事件', '互斥事件与互斥事件的区别与联系', '几何弦', '等差数列定义应用', '代数分数', '代数奇数', '频率与概率', '几何圆心角', '利用余弦定理解三角形', '绝对值证明', '已知两边及其中一边的对角解三角形', '函数性质', '全排列数公式', '平均分组无归属问题', '全排列', '平行直线方程', '代数约数', '点与圆位置关系', '不等式证明', '几何扇形', '杨辉三角', '调序法', '函数夹逼准则', '解三角形应用', '二项式定理性质', '分式增根', '条件概率应用题', '代数质因数', '等差数列的证明方法', '不等式性质与命题真假', '几何正三角形', '位置关系讨论', '分式定义', '函数渐近线', '三角形面积问题', '几何中心对称', '构造差式与合式', '点斜式方程', '几何分布期望', '两直线交点坐标', '几何矩形', '重要结论', '分组求和方法', '期望决策', '几何梯形', '几何相交弦', '隔板法', '二次不等式解法', '基本不等式', '非完全平均分组无归属问题', '二项式性质', '函数对称性', '观察法', '贝叶斯公式', '指数', '空间点坐标', '求数列的最大项问题', '函数的基本性质-单调性', '绝对值定义', '抽奖次序问题', '系数最值', '数学归纳法应用', '排列组合应用', '绝对值不等式平方法', '等比数列应用', '不等式变形', '两直线夹角', '指数函数', '等差数列前n项和的最值', '代数公因数', '轴对称性质', '三角形的面积公式', '函数抛物线', '解三角形的概念', '最值定理简记', '分式倍数', '求$S_{n}$ 的最值', '截距式方程', '直线到直线的角度', '分式解方程', '直线与圆位置关系', '将递推数列转化为等比数列的方法（待定系数法）', '两点式方程', '交点与面积问题', '函数的零点与方程', '排列定义', '几何三角函数', '两直线位置关系', '最值问题', '距离公式证明', '等差数列应用', '几何定理面积', '互斥事件', '方程列方程', '指定元素排列组合问题', '中点坐标公式', '概率的乘法公式定义', '两组相同元素的排列', '正弦定理应用', '方程解方程', '平行线间距离', '几何定理三角形内角和', '一般式方程', '对三角形解的个数的探究', '已知两角和任意一边解三角形', '直线到两个定点距离', '离散分布期望', '期望定义', '几何内角和', '等比数列与等差数列关系', '二项分布应用', '代数小数', '等比数列定义', '点斜式方程例题', '对称曲线方程', '几何定理周长', '二项式相关结论', '几何定义', '几何平行', '分式问题类型', '函数的表示法', '倾斜角性质', '不等式运算性质', '占位法', '正弦定理的表示', '余弦定理及其推论的表示', '几何平行四边形', '递推公式', '微分中值定理（罗尔定理）', '实际应用', '实际情景应用', '从函数的观点来看等差数列', '求数列的通项', '反函数', '两直线平行判断', '组合数', '几何相似三角形', '直线过定点问题', '等比数列证明', '等差中项', '空间坐标定义', '函数二元一次方程', '角平分线性质', '正弦定理的常见变形', '因式相乘型', '倾斜角范围', '函数中点坐标公式', '点到直线距离例题', '等差数列定义', '分式实际应用', '古典概型定义', '分式根与系数', '代数偶数', '代数奇数、偶数', '倒序相加法', '分段讨论法解不等式', '最值定理应用', '严格不等式', '函数的概念', '几何弧长公式', '分式通分', '几何勾股定理', '几何长方形', '几何圆锥', '方程定义', '最值定理', '待定系数法', '平均法', '特殊排列方法', '恒成立问题', '并项求和法', '几何正n边形', '点与圆位置判断', '二项式系数特点', '圆与圆位置判断', '圆的一般方程', '代数等式', '分式约分', '代数单位换算', '最值求解技巧', '截距问题', '函数导数', '几何定理体积', '组合数公式', '一般式位置关系', '等差数列性质', '组合定义', '代数倍数', '独立性性质', '几何补角', '代数分数运算', '含绝对值不等式的性质', '不等式性质的应用', '期望', '轨迹方程', '等比数列的判定', '导数的运算', '代数质数', '几何数列公式', '方程一元二次方程', '几何垂直平分线', '分式运算法则']
    """},
        # {"role": "user", "content": "解方程： 5(x + 2) = 30"},
        {"role": "user", "content": prompt},
    ]
    
    # 使用 tokenizer 处理输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    input_ids = model_inputs.input_ids
    
    max_new_tokens = 1024
    
    # 使用 generate 一次性生成
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # 使用贪婪解码
    )
    
    # 解码输出
    output_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # 使用 print 输出最终结果
    print(output_text)
    
    def base64_to_chinese(encoded_text: str) -> str:
        """将Base64编码转换回中文字符串"""
        base64_bytes = encoded_text.encode('ascii')
        utf8_bytes = base64.b64decode(base64_bytes)
        return utf8_bytes.decode('utf-8')
    
    
    def load_knowledge(kb_path: str) -> List[Dict]:
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        # decode content fields
        for entry in kb:
            if 'content' in entry and entry.get('content'):
                entry['content_view'] = base64_to_chinese(entry['content'])
        return kb
    
    
    def build_embeddings(
        kb: List[Dict],
        model: SentenceTransformer,
        embed_path: str
    ) -> np.ndarray:
        texts = []
        for entry in kb:
            combined = f"{entry['category']} [SEP] {entry['sub_category']} [SEP] {entry['content_view']}"
            texts.append(combined)
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        np.save(embed_path, embeddings)
        return embeddings
    
    
    def load_embeddings(embed_path: str) -> np.ndarray:
        return np.load(embed_path)
    
    
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # normalize
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b)
        return np.dot(a_norm, b_norm)
    
    
    def retrieve_top_k(
        kb: List[Dict],
        embeddings: np.ndarray,
        query: str,
        model: SentenceTransformer,
        threshold: float,
        labels: List[str],
        top_k: int
    ) -> List[Dict]:
        # embed query
        q_emb = model.encode(query, convert_to_numpy=True)
        sims = cosine_sim(embeddings, q_emb)
        # collect and filter
        results = []
        for idx, score in enumerate(sims):
            entry = kb[idx]
            if entry['label'] in labels and score >= threshold:
                res = entry.copy()
                del  res['content_view']
                res['score'] = float(score)
                results.append(res)
        # sort and take top_k
        
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        return results
    
    
    
    query = output_text + " " + prompt
    
    # Load or build embeddings
    kb = load_knowledge("knowledge.json")
    if os.path.exists("kb_embeddings.npy"):
        embeddings = load_embeddings("kb_embeddings.npy")
    else:
        sbert_model = SentenceTransformer('/home/hzw/.cache/ragmodel/all-MiniLM-L6-v2')
        embeddings = build_embeddings(kb, sbert_model, "kb_embeddings.npy")
    
    # Load SBERT model for query encoding
    sbert_model = SentenceTransformer('/home/hzw/.cache/ragmodel/all-MiniLM-L6-v2')
    # 学术标签：定理知识 + 解题技巧
    labels_academic = ['定理知识', '解题技巧']
    academic_res = retrieve_top_k(
        kb, embeddings, query, sbert_model,
        0.4, labels_academic, 3
    )
    print("[Academic Top Results]")
    print(json.dumps(academic_res, ensure_ascii=False, indent=2))
    
    # 题 类标签
    labels_problem = ['题']
    problem_res = retrieve_top_k(
        kb, embeddings, query, sbert_model,
        0.5, labels_problem, 3
    )
    print("[Problem Top Results]")
    print(json.dumps(problem_res, ensure_ascii=False, indent=2))
    return jsonify({'li' : [academic_res, problem_res]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
