import os
import base64
from openai import OpenAI
from typing import List
from dotenv import load_dotenv

import json

def save_result_to_json(raw_text: str, output_path="output/questions_result.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        # 尝试将模型输出解析为 JSON 格式
        parsed = json.loads(raw_text)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print(f"✅ 已成功保存结构化 JSON 至: {output_path}")
    except json.JSONDecodeError:
        # 如果不是严格 JSON 格式，保存为字符串
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"raw_text": raw_text}, f, indent=2, ensure_ascii=False)
        print(f"⚠️ 输出不是有效 JSON，已以原始文本形式保存至: {output_path}")

def save_meta_json(input_path="output/questions_result.json", output_dir="output/meta"):
    # 读取原始结构化题目 JSON
    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict) and "raw_text" in data:
                raise ValueError("❌ 原始内容不是结构化 JSON，只有纯文本")
        except Exception as e:
            print(f"❌ 无法读取或解析 JSON 文件: {e}")
            return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历题目列表，逐题保存
    for item in data:
        qid_raw = item.get("question_id", "").strip()
        qtext = item.get("question_content", "").strip()

        # 提取题号（保留数字部分，例如 "1." -> 1）
        try:
            qid = int(''.join(c for c in qid_raw if c.isdigit()))
        except:
            print(f"⚠️ 无法识别题号: {qid_raw}，跳过该题")
            continue

        out_path = os.path.join(output_dir, f"question_{qid}.json")
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump({"question_id": qid, "question_content": qtext}, f_out, ensure_ascii=False, indent=2)

    print(f"✅ 已成功分割并保存为 {len(data)} 个题目文件至: {output_dir}")

# 初始化 GPT-4o 客户端
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)  # ✅ 强制覆盖
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 将本地图片转为 base64
def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 输入图片路径列表
image_paths = [
    "AAAAAA/卷2_01.png",
    "AAAAAA/卷2_02.png",
    "AAAAAA/卷2_03.png",
    "AAAAAA/卷2_04.png"
]

# 构造 GPT-4o 的 messages 格式
def build_gpt4o_input(images: List[str]):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "你是一个OCR结构化信息提取器，我会给你一整张试卷（由多张图片组成），"
                        "请你识别每一道题目，提取其题号（例如'1.'、'2.'）和题干内容（包括所有题目的文字,所有的选项。如果有图片或表格的话，对其内容生成文字描述。），"
                        "要求输出格式如下：每道题是一个对象，包含question_id 和 question_content。"
                    )
                }
            ] + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_to_base64(p)}"
                    }
                } for p in images
            ]
        }
    ]
    return messages

# GPT-4o 提取题目结构
def extract_questions(messages):
    response = client.chat.completions.create(
        model="o4-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()



# ========= 主逻辑 =========

if __name__ == "__main__":
    
    print("📄 提取结构中...")
    msg1 = build_gpt4o_input(image_paths)
    result1 = extract_questions(msg1)
    
    # 提取完成后调用保存
    save_result_to_json(result1)
    save_meta_json()

    print("\n✅ 结果：\n")
    print(result1)