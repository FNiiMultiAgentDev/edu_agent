# corrector.py
import os, json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.nuwaapi.com/v1"
)

# 数据模型
class PointsEarnedAndWhy(BaseModel):
    points_earned_of_this_point: int
    why: str

class CorrectionAndExplanation(BaseModel):
    point_1: PointsEarnedAndWhy
    point_2: PointsEarnedAndWhy
    point_3: PointsEarnedAndWhy
    point_4: PointsEarnedAndWhy
    point_5: PointsEarnedAndWhy

class QuestionGrading(BaseModel):
    points_earned_of_this_question: float
    correction_and_explanation: CorrectionAndExplanation

# 主函数
def correct(student_answer: str):
    # 读取静态信息
    with open("zhengzhi/question.txt", encoding='utf-8') as f:
        question = f.read()
    with open("zhengzhi/ground_truth_answer.txt", encoding='utf-8') as f:
        ground_truth = f.read()
    with open("zhengzhi/explanation_of_ground_truth_answer_path.txt", encoding='utf-8') as f:
        explanation = f.read()
    with open("zhengzhi/few_shot_example.json", encoding='utf-8') as f:
        few_shot = json.load(f)

    # 调用 API
    response = client.responses.parse(
        model="o4-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "你是一名严谨的高中政治老师，请根据评分标准批改学生答案。批改时要分点批改，每一点为2分，可能得分为0/2，1/2，2/2。"},
                {"type": "input_text", "text": f"题目：{question}"},
                {"type": "input_text", "text": f"标准答案：{ground_truth}"},
                {"type": "input_text", "text": f"题目解析与各点评分标准：{explanation}"},
                {"type": "input_text", "text": f"批改示例：{few_shot}"},
                {"type": "input_text", "text": f"学生答案：{student_answer}"},
            ],
        }],
        text_format=QuestionGrading,
    )

    # 返回结构化评分结果
    return response.output_parsed
