import os
import json
import base64
import argparse
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer


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
            res['score'] = float(score)
            results.append(res)
    # sort and take top_k
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    return results


def main():
    parser = argparse.ArgumentParser(description='RAG Retriever with separate thresholds and top-k')
    parser.add_argument('--kb', type=str, required=True, help='知识库 JSON 文件路径')
    parser.add_argument('--embed', type=str, default='kb_embeddings.npy', help='存储 Embeddings 的文件')
    parser.add_argument('--threshold_academic', type=float, default=0.7, help='定理知识/解题技巧 阈值')
    parser.add_argument('--threshold_problem', type=float, default=0.7, help='题 阈值')
    parser.add_argument('--top_k', type=int, default=3, help='每类返回的最大条数')
    parser.add_argument('--query', type=str, required=True, help='用户查询文本')
    args = parser.parse_args()

    # Load or build embeddings
    kb = load_knowledge(args.kb)
    if os.path.exists(args.embed):
        embeddings = load_embeddings(args.embed)
    else:
        sbert_model = SentenceTransformer('/home/hzw/.cache/ragmodel/all-MiniLM-L6-v2')
        embeddings = build_embeddings(kb, sbert_model, args.embed)

    # Load SBERT model for query encoding
    sbert_model = SentenceTransformer('/home/hzw/.cache/ragmodel/all-MiniLM-L6-v2')
    # 学术标签：定理知识 + 解题技巧
    labels_academic = ['定理知识', '解题技巧']
    academic_res = retrieve_top_k(
        kb, embeddings, args.query, sbert_model,
        args.threshold_academic, labels_academic, args.top_k
    )
    print("[Academic Top Results]")
    print(json.dumps(academic_res, ensure_ascii=False, indent=2))

    # 题 类标签
    labels_problem = ['题']
    problem_res = retrieve_top_k(
        kb, embeddings, args.query, sbert_model,
        args.threshold_problem, labels_problem, args.top_k
    )
    print("[Problem Top Results]")
    print(json.dumps(problem_res, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

