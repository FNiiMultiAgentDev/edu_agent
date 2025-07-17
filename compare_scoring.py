#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from typing import Dict, Any

def load_json(path: str) -> Dict[str, Any]:
    """从指定路径加载 JSON 文件并返回解析后的字典。"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calc_accuracy(human: Dict[str, Any], model: Dict[str, Any]) -> float:
    """
    比较 human（人工批改）和 model（大模型批改）两个结果中各“得分点”的 '该点得分'。
    返回准确率：匹配得分点数 / 总得分点数。
    """
    human_exp = human.get("explanation", {})
    model_exp = model.get("explanation", {})

    pts = list(human_exp.keys())
    if not pts:
        return 0.0

    correct = 0
    for pt in pts:
        human_score = human_exp[pt].get("该点得分")
        model_score = model_exp.get(pt, {}).get("该点得分")
        if human_score == model_score:
            correct += 1

    return correct / len(pts)

def main(ref_path: str, model_path: str):
    # 1. 加载“标准 + 人工批改”参考 JSON，以及“大模型批改” JSON
    ref_data   = load_json(ref_path).get("scoring_rules", {})
    model_data = load_json(model_path).get("scoring_rules", {})

    # 2. 遍历所有 wrong_N 实例，计算准确率
    accuracies = {}
    for key, human_res in ref_data.items():
        if key == "correct":
            continue  
        if key not in model_data:
            print(f"WARNING: 模型结果中缺少实例 '{key}'，跳过比较。", file=sys.stderr)
            continue
        model_res = model_data[key]
        acc = calc_accuracy(human_res, model_res)
        accuracies[key] = acc
        print(f"{key} 准确率: {acc * 100:.1f}%")

    # 3. 计算并输出总体准确率
    if accuracies:
        overall = sum(accuracies.values()) / len(accuracies)
        print(f"\n准确率: {overall * 100:.1f}%")
    else:
        print("未找到任何可比较的实例。", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python compare_scoring.py <参考JSON路径> <模型JSON路径>", file=sys.stderr)
        sys.exit(1)
    ref_json, model_json = sys.argv[1], sys.argv[2]
    main(ref_json, model_json)
