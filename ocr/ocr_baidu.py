import re
import numpy as np
import sys
import os
import json
from paddleocr import PaddleOCR
from difflib import SequenceMatcher
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zhengzhi.corrector import correct

def get_center_y(box):
    return (box[0][1] + box[3][1]) / 2

def group_by_lines(char_list, y_thresh=15):
    """将字符按Y坐标聚类为多行"""
    lines = []
    for ch in char_list:
        cy = get_center_y(ch['box'])
        matched = False
        for line in lines:
            line_cy = get_center_y(line[0]['box'])
            if abs(cy - line_cy) < y_thresh:
                line.append(ch)
                matched = True
                break
        if not matched:
            lines.append([ch])
    return lines

def build_colored_path_from_lines(line_groups):
    """将多行字符构造成一个连续染色区域（首尾字符拼接）"""
    path_points = []
    for line in line_groups:
        first = line[0]['box']
        last = line[-1]['box']
        tl = first[0]
        bl = first[3]
        tr = last[1]
        br = last[2]
        path_points.extend([tl, tr, br, bl])
    return [list(map(int, pt)) for pt in path_points]

# 初始化 OCR
ocr = PaddleOCR(use_textline_orientation=True, lang='ch')
res = ocr.predict("ocr/input.jpg")[0]

texts = res['rec_texts']
scores = res['rec_scores']
boxes = res['rec_polys']

# 构建字符级序列
char_stream = []
for text, score, box in zip(texts, scores, boxes):
    chars = list(text.strip())
    n = len(chars)
    box = np.array(box)
    top_line = np.linspace(box[0], box[1], n + 1)
    bottom_line = np.linspace(box[3], box[2], n + 1)
    for i, ch in enumerate(chars):
        tl = top_line[i]
        tr = top_line[i+1]
        br = bottom_line[i+1]
        bl = bottom_line[i]
        char_box = np.array([tl, tr, br, bl], dtype=int).tolist()
        char_stream.append({'char': ch, 'box': char_box, 'score': score})

#print(char_stream[0])

# 构建全文
full_text = ''.join([c['char'] for c in char_stream])

student_answer = full_text
with open("ocr/student_answer.txt", "w", encoding="utf-8") as f:
    f.write(student_answer)
# 调用作业批改接口
grading_result = correct(student_answer)

# 切分为句子（排除顿号）
split_pattern = re.compile(r"([，。！？；：. ① ② ③ ④ ⑤])")
segments = []
start_idx = 0
for match in split_pattern.finditer(full_text):
    end_idx = match.end()
    segment_text = full_text[start_idx:end_idx]
    segment_chars = char_stream[start_idx:end_idx]
    line_groups = group_by_lines(segment_chars)
    polygon_path = build_colored_path_from_lines(line_groups)
    segments.append({
        'text': segment_text,
        'box': polygon_path,
        'score': np.mean([c['score'] for c in segment_chars])
    })
    start_idx = end_idx

#处理结尾残余
if start_idx < len(full_text):
    segment_chars = char_stream[start_idx:]
    segment_text = full_text[start_idx:]
    line_groups = group_by_lines(segment_chars)
    polygon_path = build_colored_path_from_lines(line_groups)
    segments.append({
        'text': segment_text,
        'box': polygon_path,
        'score': np.mean([c['score'] for c in segment_chars])
    })

#######################################################
###两两组合、三三组合

# 存放所有组合后的新 segment
combined_segments = []

# 相邻两两组合
for i in range(len(segments) - 1):
    seg1 = segments[i]
    seg2 = segments[i + 1]
    combined_text = seg1['text'] + seg2['text']
    combined_box = seg1['box'] + seg2['box']
    combined_score = (seg1['score'] + seg2['score']) / 2
    combined_segments.append({
        'text': combined_text,
        'box': combined_box,
        'score': combined_score
    })

# 相邻三段组合
for i in range(len(segments) - 2):
    seg1 = segments[i]
    seg2 = segments[i + 1]
    seg3 = segments[i + 2]
    combined_text = seg1['text'] + seg2['text'] + seg3['text']
    combined_box = seg1['box'] + seg2['box'] + seg3['box']
    combined_score = (seg1['score'] + seg2['score'] + seg3['score']) / 3
    combined_segments.append({
        'text': combined_text,
        'box': combined_box,
        'score': combined_score
    })

# 把组合段加到 segments 里统一处理
segments += combined_segments
#######################################################

with open("ocr/target_list.txt", "r", encoding="utf-8") as f:
    target_list = [line.strip() for line in f if line.strip()]

highlight_boxes = []

# 遍历所有 target
for target in target_list:
    for item in segments:
        sim = SequenceMatcher(None, item['text'], target).ratio()
        item['sim'] = sim
        if sim >= 0.8:
            for point in item['box']:
                highlight_boxes.append(point)

from PIL import Image, ImageDraw

# 加载原图（确保是RGBA）
base = Image.open("ocr/input.jpg").convert("RGBA")

# 创建一个透明图层
overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

# 绘制半透明红色多边形到 overlay上
box = highlight_boxes
for i in range(0, len(box), 4):
    draw.polygon(box[i:i+4], fill=(0, 225, 0, 100))  # 每4个点为一组多边形

# 合成原图与染色图层
out = Image.alpha_composite(base, overlay)

# 保存最终效果
out.save("ocr/output_baidu.png")
print("✅ 染色结果已更新为半透明红色并保存为 output_baidu.png")

# 将 grading_result 转为 dict（如果是 Pydantic 模型）
if hasattr(grading_result, "model_dump"):
    grading_dict = grading_result.model_dump()
elif hasattr(grading_result, "dict"):
    grading_dict = grading_result.dict()
else:
    grading_dict = grading_result  # 如果它本来就是字典结构

# 添加总得分字段（如果是从外部拿到的）
grading_dict["points_earned_of_this_question"] = getattr(grading_result, "points_earned_of_this_question", None)

# 打印 JSON
print("🎯 作业批改结果（JSON 格式）：")
print(json.dumps(grading_dict, ensure_ascii=False, indent=2))

with open("ocr/grading_result.json", "w", encoding="utf-8") as f:
    json.dump(grading_dict, f, ensure_ascii=False, indent=2)
print("✅ 作业批改结果已保存为 ocr/grading_result.json")
