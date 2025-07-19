import json
import base64
import os
from pathlib import Path
from openai import OpenAI
from typing import Tuple, List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import median
from collections import defaultdict
import datetime
import copy
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EduAgent:
    def __init__(self, 
                 students_root: str = "students", 
                 questions_root: str = "questions",
                 exam_id: str = "exam_5",
                 grade: str = "高三",
                 class_id: str = "5班",
                 paper_id: str = "第二学期 第五次考试",
                 current_exam_number: int = 5,
                 openai_api_key: str = os.getenv("DASHSCOPE_API_KEY"),
                 openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.students_root = students_root
        self.questions_root = questions_root
        self.exam_id = exam_id
        self.grade = grade
        self.class_id = class_id
        self.paper_id = paper_id
        self.current_exam_number = current_exam_number
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url
        )

    # ========== 评分相关 ==========
    def scoring_single_choice(self,question_id: str, student_s_answer: str)-> Tuple[int, str, bool]:
        """
        为单项选择题进行评分
        
        Args:
            question_id (str): 题目ID，用于定位题目元数据文件。默认值为 "1"
            student_answer (str): 学生的答案选项（如 "A", "B", "C", "D"）。默认值为 "A"
        
        Returns:
            Tuple[int, str, bool]: 返回三元组
                - point_earned: 学生获得的分数
                - question_focus: 题目考查重点/知识点
                - get_full_point: 是否获得满分（布尔值）
        """
        question_metadata_path = f"questions/{question_id}/question_metadata.json"
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            correct_answer = json.load(question_metadata)["correct_answer"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            full_point = json.load(question_metadata)["full_point"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            question_focus = json.load(question_metadata)["question_focus"]
        if correct_answer == student_s_answer:
            point_earned = full_point
        else:
            point_earned = 0
        get_full_point = (point_earned == full_point)
        return point_earned, question_focus, get_full_point

    def scoring_multiple_choice(self,question_id = str, student_s_answer = str)-> Tuple[int, str, bool]:
        """
        为多项选择题进行评分
        
        Args:
            question_id (str): 题目ID，用于定位题目元数据文件。默认值为 "9"
            student_answer (str): 学生的答案选项（如 "ABC", "B", "C", "D"）。默认值为 "A"
        
        Returns:
            Tuple[int, str, bool]: 返回三元组
                - point_earned: 学生获得的分数
                - question_focus: 题目考查重点/知识点
                - get_full_point: 是否获得满分（布尔值）
        """
        question_metadata_path = f"questions/{question_id}/question_metadata.json"
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            correct_answer = json.load(question_metadata)["correct_answer"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            full_point = json.load(question_metadata)["full_point"]
        with open(question_metadata_path,
                "r", encoding="utf-8") as question_metadata:
            partial_correct_point = json.load(question_metadata)["scoring_rules"]["partially_correct"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            question_focus = json.load(question_metadata)["question_focus"]
        if correct_answer == student_s_answer:
            point_earned = full_point
        elif not (set(["A", "B", "C", "D"]) - set(list(correct_answer))) & set(list(student_s_answer)):
            point_earned = partial_correct_point
        else:
            point_earned = 0
        get_full_point = (point_earned == full_point)
        return point_earned, question_focus, get_full_point

    def scoring_fill_in_blank(self,question_id = str, student_s_answer = list[str])-> Tuple[int, str, bool]:
        """
        为填空题进行评分
        
        Args:
            question_id (str): 题目ID，用于定位题目元数据文件。默认值为 "12"
            student_answer (str): 学生的答案
        
        Returns:
            Tuple[int, str, bool]: 返回三元组
                - point_earned: 学生获得的分数
                - question_focus: 题目考查重点/知识点
                - get_full_point: 是否获得满分（布尔值）
        """
        def compare_difference(correct_answer: list, student_s_answer: list)-> np.array:
            """
            统计列表student_s_answer中与correct_answer对应位置元素不同的数量

            参数:
                correct_answer (list): 列表存储的参考答案
                student_s_answer (list): 列表存储的学生答案

            返回:
                np.array: 每个位置是否元素相等
            """
            if len(correct_answer) != len(student_s_answer):
                raise ValueError("提取出的列表存储的学生答案长度和参考答案列表长度不匹配，转交给大模型批改，请稍等")
            return np.array([a != b for a, b in zip(correct_answer, student_s_answer)])
        question_metadata_path = f"questions/{question_id}/question_metadata.json"
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            correct_answer = json.load(question_metadata)["correct_answer"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            scoring_rules = json.load(question_metadata)["scoring_rules"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            full_point = json.load(question_metadata)["full_point"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            question_focus = json.load(question_metadata)["question_focus"]
        difference_comparison_between_correct_answer_and_student_s_answer = compare_difference(correct_answer,student_s_answer)
        point_earned = full_point + difference_comparison_between_correct_answer_and_student_s_answer@np.array(scoring_rules)
        point_earned = int(point_earned)    
        get_full_point = (point_earned == full_point)
        return point_earned, question_focus, get_full_point

    def scoring_comprehensive_problems(self,question_id: str, student_s_answer: str)-> Tuple[int, str, bool, str]:
        """
        为解答题进行评分
        
        Args:
            question_id (str): 题目ID，用于定位题目元数据文件。
            student_answer (str): 学生的答案
        
        Returns:
            Tuple[int, str, bool, str]: 返回四元组
                - point_earned: 学生获得的分数
                - question_focus: 题目考查重点/知识点
                - get_full_point: 是否获得满分（布尔值）
                - LLM_feedback: 大模型批改结果
        """
        question_metadata_path = f"questions/{question_id}/question_metadata.json"
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            question = json.load(question_metadata)["question"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            full_point = json.load(question_metadata)["full_point"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            question_focus = json.load(question_metadata)["question_focus"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            grading_rubric = json.load(question_metadata)["grading_rubric"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            few_shot = json.load(question_metadata)["few_shot"]

        client = OpenAI(
            api_key="sk-9645af39a0e44b1fa14e73d68a4b9b68",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": f"""现在你是一个中学老师，你要负责批改你学生的数学试卷的题目，
                    主任要求你严格按照题目解析与各点评分标准里的踩分点来进行批改得分，
                    并对每一个踩分点进行解释，比如说这个踩分点有对应的公式，得到相应的分数，不是累计得分！，
                    那个踩分点没有公式或者公式错误，不得分"。按照批改示例里的"few_shot_output”的json格式返回一个{{LLM_feedback:{{"correction_and_explanation":{{}}}},"point_earned_of_this_question":}}，遵循few_shot_output里的给分点:
                    题目：{question}
                    评分细则:{grading_rubric}
                    批改示例：{few_shot}
                """
                },
                {
                    "role": "user",
                    "content": f"学生答案：{student_s_answer}", 
                },
            ],
            temperature= 0.3,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False}
        )

        json_string = completion.choices[0].message.content
        json_string = json.loads(json_string)
        print(json_string)

        LLM_feedback = json_string["LLM_feedback"]
        point_earned = json_string["LLM_feedback"]["point_earned_of_this_question"]
        get_full_point = (point_earned == full_point)
        
        return point_earned, question_focus, get_full_point, LLM_feedback
    
    def scoring_by_LLM_without_answer(self,question_id: str, student_s_answer: str):
        """
        大模型判断学生答案是否正确的函数，适用于只缺答案和评分标准的情况
        
        Args:
            question_id (str): 题目ID，用于定位题目元数据文件。
            student_answer (str): 学生的答案
        
        Returns:
            Tuple[int, str, bool, str]: 返回四元组
                - point_earned: 学生获得的分数
                - question_focus: 题目考查重点/知识点
                - get_full_point: 是否获得满分（布尔值）
                - LLM_feedback: 大模型批改结果
        """
        question_metadata_path = f"questions/{question_id}/question_metadata.json"
        
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            question = json.load(question_metadata)["question"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            question_focus = json.load(question_metadata)["question_focus"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            full_point = json.load(question_metadata)["full_point"]
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            section_it_belongs_to = json.load(question_metadata)["section_it_belongs_to"]
        scoring_rules_given_question_section_path = f"scoring_rules_given_question_section/{section_it_belongs_to}.json"
        with open(scoring_rules_given_question_section_path, "r", encoding="utf-8") as scoring_rules_given_question_section:
            scoring_rules = json.load(scoring_rules_given_question_section)["scoring_description"]
        client = OpenAI(
            api_key="sk-9645af39a0e44b1fa14e73d68a4b9b68",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": f"""现在你是一个高中老师，你要负责批改你学生的数学试卷的题目，
                    主任要求你判断学生的答案是否正确，应该给几分  
                    题目：{question}
                    
                """
                },
                {
                    "role": "user",
                    "content": f"学生答案：{student_s_answer}", 
                },
            ],
            temperature= 0.3,
            extra_body={"enable_thinking": True},
            stream=True,
        
        )
        reasoning_content = ""  # 完整思考过程
        answer_content = ""  # 完整回复
        is_answering = False  # 是否进入回复阶段
        #print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

        for chunk in completion:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
                continue

            delta = chunk.choices[0].delta

            # 只收集思考内容
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    #print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

            # 收到content，开始进行回复
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    #print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                #print(delta.content, end="", flush=True)
                answer_content += delta.content
        LLM_feedback = "大模型评分:"+answer_content+"\n\n思考过程：" + reasoning_content
        completion = client.chat.completions.create(
            model="qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": f"""现在你是一个高中老师，你要负责批改你学生的数学试卷的题目，
                这是大模型对于学生这道题答题情况的分析：{LLM_feedback}。
                这是这道题满分{full_point}分，请你判断这道题学生应该得几分，返回如下json格式：{{"point_earned": <学生获得的分数,int>, "get_full_point": <是否获得满分,bool:True/False>, }}。  
                
                """
                },
                {
                    "role": "user",
                    "content": f"学生答案：{student_s_answer}", 
                },
            ],
            temperature= 0.3,
            extra_body={"enable_thinking": False},
            response_format={"type": "json_object"},
        )
        print(completion.choices[0].message.content)
        print(type(completion.choices[0].message.content))
        point_earned = json.loads(completion.choices[0].message.content)["point_earned"]
        get_full_point = json.loads(completion.choices[0].message.content)["get_full_point"]

        return point_earned, question_focus, get_full_point, LLM_feedback
    
    def merely_get_wrong_or_correct_from_LLM(self,question_id: str, student_s_answer: str):
        """
        大模型判断学生答案是否正确的函数，适用于只有题目的情况
        
        Args:
            question_id (str): 题目ID，用于定位题目元数据文件。
            student_answer (str): 学生的答案
        
        Returns:
            LLM_feedback (str): 大模型批改结果
        """
        question_metadata_path = f"questions/{question_id}/question_metadata.json"
        with open(question_metadata_path, "r", encoding="utf-8") as question_metadata:
            question = json.load(question_metadata)["question"]

        client = OpenAI(
            api_key="sk-9645af39a0e44b1fa14e73d68a4b9b68",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": f"""现在你是一个中学老师，你要负责批改你学生的数学试卷的题目，
                    主任要求你判断学生的答案是否正确。  
                    题目：{question}
                """
                },
                {
                    "role": "user",
                    "content": f"学生答案：{student_s_answer}", 
                },
            ],
            temperature= 0.3,
            extra_body={"enable_thinking": True},
            stream=True,
        
        )
        reasoning_content = ""  # 完整思考过程
        answer_content = ""  # 完整回复
        is_answering = False  # 是否进入回复阶段
        #print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

        for chunk in completion:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
                continue

            delta = chunk.choices[0].delta

            # 只收集思考内容
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    #print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

            # 收到content，开始进行回复
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    #print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                #print(delta.content, end="", flush=True)
                answer_content += delta.content
        LLM_feedback = "大模型评分"+answer_content+"\n\n思考过程：" + reasoning_content
            
        print(LLM_feedback)

        return LLM_feedback
    # ========== 学生答案提取 ==========
    def student_s_answers_extractor(self,student_id = 2024001):
        def save_json_string_to_file(json_string, file_path):
            """
            将JSON字符串保存为本地JSON文件
            
            Args:
                json_string (str): JSON格式的字符串
                file_path (str): 要保存的文件路径
            """
            try:
                # 首先验证JSON字符串是否有效
                json_data = json.loads(json_string)
                
                # 写入文件，使用ensure_ascii=False支持中文字符
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                print(f"JSON文件已成功保存到: {file_path}")
                
            except json.JSONDecodeError as e:
                print(f"JSON格式错误: {e}")
            except Exception as e:
                print(f"保存文件时出错: {e}")
    #  base 64 编码格式
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        base64_image_1 = encode_image(f"../edu_agent/students/{student_id}/answered_paper_sheet_scan/1.jpg")
        base64_image_2 = encode_image(f"../edu_agent/students/{student_id}/answered_paper_sheet_scan/2.jpg")
        base64_image_3 = encode_image(f"../edu_agent/students/{student_id}/answered_paper_sheet_scan/3.jpg")

        client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        few_shot = {"单项选择题":{"1":"A","2":"B"},"多项选择题":{"9":"ABC"},"填空题":{"12":[""],"13":[""],"14":[""]},"解答题":{"15":"忠实转录学生的手写解题过程"}}
        student_s_answers = client.chat.completions.create(
            model="qwen-vl-max-latest", 
            messages=[
                {
                    "role": "system",
                    "content": [{"type":"text","text":f"你是一台作业扫描机，请你按照顺序把学生手写的答案提取为这样的格式：{few_shot}，不要篡改、添加、删减学生的手写内容"}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image_1}"}, 
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image_2}"}, 
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image_3}"}, 
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )
        print(student_s_answers.choices[0].message.content)
        save_json_string_to_file(student_s_answers.choices[0].message.content, file_path = f"../edu_agent/students/{student_id}/student_s_answers.json")
        return student_s_answers.choices[0].message.content

    # ========== 批改执行 ==========
    def scoring_executor(self,student_id = 2024001):
        # 读取JSON文件
        student_answer_revise_path = f"../edu_agent/students/{student_id}/answer_revise.json"
        student_s_answers_path = f"../edu_agent/students/{student_id}/student_s_answers.json"
        with open(student_answer_revise_path, 'r', encoding='utf-8') as file:
            answer_revise = json.load(file)
        with open(student_s_answers_path, 'r', encoding='utf-8') as file:
            student_s_answers = json.load(file)
        for idx, question_id in enumerate(student_s_answers["单项选择题"]):
            LLM_feedback = ""
            with open(f"../edu_agent/questions/{question_id}/question_metadata.json", 'r', encoding='utf-8') as file:
                correct_answer = json.load(file)["correct_answer"]
            with open(f"../edu_agent/questions/{question_id}/question_metadata.json", 'r', encoding='utf-8') as file:
                grading_rubric = json.load(file).get("grading_rubric")
            if (correct_answer != None) | (grading_rubric != None):
                point_earned, question_focus, get_full_point = self.scoring_single_choice(question_id, student_s_answers["单项选择题"][question_id])
            else:
                point_earned, question_focus, get_full_point, LLM_feedback = self.scoring_by_LLM_without_answer(question_id, student_s_answers["单项选择题"][question_id])
            # 修改现有键的值
            answer_revise[question_id] = {"point_earned" : point_earned,
                                        "question_focus" : question_focus,
                                        "get_full_point" : get_full_point,
                                        "LLM_feedback": LLM_feedback
                                        }
            
        for idx, question_id in enumerate(student_s_answers["多项选择题"]):
            LLM_feedback = ""
            with open(f"../edu_agent/questions/{question_id}/question_metadata.json", 'r', encoding='utf-8') as file:
                correct_answer = json.load(file)["correct_answer"]
            with open(f"../edu_agent/questions/{question_id}/question_metadata.json", 'r', encoding='utf-8') as file:
                grading_rubric = json.load(file).get("grading_rubric")
            if (correct_answer != None) | (grading_rubric != None):
                point_earned, question_focus, get_full_point = self.scoring_multiple_choice(question_id, student_s_answers["多项选择题"][question_id])
            else:
                point_earned, question_focus, get_full_point, LLM_feedback = self.scoring_by_LLM_without_answer(question_id, student_s_answers["多项选择题"][question_id])
            # 修改现有键的值
            answer_revise[question_id] = {"point_earned" : point_earned,
                                        "question_focus" : question_focus,
                                        "get_full_point" : get_full_point,
                                        "LLM_feedback": LLM_feedback
                                        }
            
        for idx, question_id in enumerate(student_s_answers["填空题"]):
            LLM_feedback = ""
            with open(f"../edu_agent/questions/{question_id}/question_metadata.json", 'r', encoding='utf-8') as file:
                correct_answer = json.load(file)["correct_answer"]
            with open(f"../edu_agent/questions/{question_id}/question_metadata.json", 'r', encoding='utf-8') as file:
                grading_rubric = json.load(file).get("grading_rubric")
            if (correct_answer != None) | (grading_rubric != None):
                point_earned, question_focus, get_full_point = self.scoring_fill_in_blank(question_id, student_s_answers["填空题"][question_id])
            else:
                point_earned, question_focus, get_full_point, LLM_feedback = self.scoring_by_LLM_without_answer(question_id, student_s_answers["填空题"][question_id])
            # 修改现有键的值
            answer_revise[question_id] = {"point_earned" : point_earned,
                                        "question_focus" : question_focus,
                                        "get_full_point" : get_full_point,
                                        "LLM_feedback": LLM_feedback
                                        }

        for idx, question_id in enumerate(student_s_answers["解答题"]):
            LLM_feedback = ""
            with open(f"../edu_agent/questions/{question_id}/question_metadata.json", 'r', encoding='utf-8') as file:
                grading_rubric = json.load(file).get("grading_rubric")
            if (correct_answer != None) | (grading_rubric != None):
                point_earned, question_focus, get_full_point, LLM_feedback = self.scoring_comprehensive_problems(question_id, student_s_answers["解答题"][question_id])
            else:
                print(f"第{question_id}题没找到现成的答案，转交大模型思考判断，请稍后")
                point_earned, question_focus, get_full_point, LLM_feedback = self.scoring_by_LLM_without_answer(question_id, student_s_answers["解答题"][question_id])
                print(f"第{question_id}题没找到现成的答案，大模型自主判断完成")
            # 修改现有键的值
            answer_revise[question_id] = {"point_earned" : point_earned,
                                        "question_focus" : question_focus,
                                        "get_full_point" : get_full_point,
                                        "LLM_feedback": LLM_feedback
                                        }
    
        # 计算总分
        total_point_earned_of_this_exam = 0
        for item in answer_revise:
            if item != "total_point_earned_of_this_exam":
                #print(answer_revise[item])
                total_point_earned_of_this_exam += answer_revise[item]["point_earned"]
        answer_revise["total_point_earned_of_this_exam"] = total_point_earned_of_this_exam
        

        print(answer_revise)
        # 写回文件
        with open(student_answer_revise_path, 'w', encoding='utf-8') as file:
            json.dump(answer_revise, file, indent=2, ensure_ascii=False)

    # ========== 排名与记录 ==========
    def generate_student_exam_rank(self):
        def _list_direct_folders(root_path):
            students_paths = []
            path = Path(root_path)
            for item in path.iterdir():
                if item.is_dir():
                    students_paths.append(item.name)
            return students_paths
        students_paths = _list_direct_folders("students")
        student_id_to_total_score = {}
        for item in students_paths:
            student_answer_revise_path = f"../edu_agent/students/{item}/answer_revise.json"
            with open(student_answer_revise_path, 'r', encoding='utf-8') as file:
                answer_revise = json.load(file)
            student_id_to_total_score[item] = answer_revise["total_point_earned_of_this_exam"]
        sorted_student_id_to_total_score = sorted(student_id_to_total_score.items(), key=lambda x: x[1], reverse=True)
        rankings = {}
        current_rank = 1
        draw = 0
        for i, (student_id, score) in enumerate(sorted_student_id_to_total_score):
            if score < sorted_student_id_to_total_score[i-1][1]:
                current_rank = i + draw
                draw = 0
            else: draw += 1    
            rankings[student_id] = current_rank
        return rankings
    
    def write_student_exam_result_into_exam_record(self,student_id_to_rank_dic = {}): 
        def _list_direct_folders(root_path):
            students_paths = []
            path = Path(root_path)
            for item in path.iterdir():
                if item.is_dir():
                    students_paths.append(item.name)
            return students_paths
        students_paths = _list_direct_folders("students")
        for item in students_paths:
            student_answer_revise_path = f"../edu_agent/students/{item}/answer_revise.json"
            student_exam_record_path = f"../edu_agent/students/{item}/exam_record.json"
            with open(student_answer_revise_path, 'r', encoding='utf-8') as file:
                answer_revise = json.load(file)
            with open(student_exam_record_path, 'r', encoding='utf-8') as file:
                exam_record = json.load(file)
            exam_record[self.exam_id] = {}
            exam_record[self.exam_id]["score"] = answer_revise["total_point_earned_of_this_exam"]
            exam_record[self.exam_id]["rank"] = student_id_to_rank_dic[item]
            wrong_focus = []
            for item in answer_revise:
                if item != "total_point_earned_of_this_exam":
                    if answer_revise[item]["get_full_point"] == False:
                        wrong_focus.append(answer_revise[item]["question_focus"])
            exam_record[self.exam_id]["wrong_focus"] = wrong_focus

            with open(student_exam_record_path, 'w', encoding='utf-8') as file:
                json.dump(exam_record, file, indent=2, ensure_ascii=False)

    # ========== 学生报告 ==========
    def generate_report(self,student_dir):
        # 读取 json 文件
        with open(os.path.join(student_dir, "answer_revise.json"), "r", encoding="utf-8") as f:
            answers = json.load(f)
        with open(os.path.join(student_dir, "student_info.json"), "r", encoding="utf-8") as f:
            info = json.load(f)
        with open(os.path.join(student_dir, "exam_record.json"), "r", encoding="utf-8") as f:
            history = json.load(f)

        # 当前考试信息
        current_exam = self.current_exam_number
        current_record = history.get(f"exam_{current_exam}")
        current_rank = current_record["rank"]
        current_score = current_record["score"]

        # 错误题目和知识点
        wrong_items = []
        for qid_str, content in answers.items():
            if not isinstance(content, dict):
                continue
            qid = int(qid_str)
            if not content.get("get_full_point", False):
                wrong_items.append((qid, content.get("question_focus", "无")))

        # 历史分数与排名（去除当前这次再手动添加，确保顺序）
        exam_nums = []
        scores = []
        ranks = []
        for key, record in history.items():
            eid = int(key.replace("exam_", ""))
            if eid != current_exam:
                exam_nums.append(eid)
                scores.append(record["score"])
                ranks.append(record["rank"])
        exam_nums.append(current_exam)
        scores.append(current_score)
        ranks.append(current_rank)
        exam_nums, scores, ranks = zip(*sorted(zip(exam_nums, scores, ranks)))

        # 文本总结
        rank_progress = ""
        if len(ranks) >= 2:
            delta = ranks[-2] - ranks[-1]
            symbol = "↑" if delta > 0 else "↓"
            rank_progress = f"与上次相比，排名{symbol}{abs(delta)}名（从第{ranks[-2]}名到第{ranks[-1]}名）"
        if len(ranks) >= 5:
            delta_total = ranks[0] - ranks[-1]
            symbol_total = "↑" if delta_total > 0 else "↓"
            rank_progress += f"，与第一次考试相比{symbol_total}{abs(delta_total)}名（从第{ranks[0]}名到第{ranks[-1]}名）"

        # 图表生成
        fig_dir = os.path.join(student_dir, "report_figures")
        os.makedirs(fig_dir, exist_ok=True)

        # 分数变化
        plt.figure()
        plt.plot(exam_nums, scores, marker="o")
        plt.title("分数变化折线图")
        plt.xlabel("考试次数")
        plt.ylabel("总分")
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "score_trend.png"))
        plt.close()

        # 排名变化（排名小在上）
        plt.figure()
        plt.plot(exam_nums, ranks, marker="o")
        plt.gca().invert_yaxis()
        plt.title("排名变化折线图（越高越靠上）")
        plt.xlabel("考试次数")
        plt.ylabel("排名")
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "rank_trend.png"))
        plt.close()

        # Markdown 生成
        md_lines = [
            f"# 📄 学生考试报告：{info['name']}\n",
            f"## 基本信息\n",
            f"- 学号：{info['student_id']}\n",
            f"- 班级：{self.fixed_grade} {self.fixed_class}\n",
            f"- 性别：{info['gender']}\n",
            f"- 考试编号：{self.fixed_paper_id}\n",
            f"- 当前总分：{current_score}，当前排名：第{current_rank}名\n",
            f"- {rank_progress}\n",
            "\n## 错误题目与知识点\n",
        ]
        if wrong_items:
            for qid, focus in wrong_items:
                md_lines.append(f"- 题目 {qid}：{focus}\n")
        else:
            md_lines.append("✅ 本次考试全部答对，真棒！\n")

        md_lines += [
            "\n## 历史分数与排名变化\n",
            "![分数变化图](report_figures/score_trend.png)\n",
            "![排名变化图](report_figures/rank_trend.png)\n",
        ]

        from openai import OpenAI
        client = OpenAI(
            api_key = "sk-10acd30d8f8c4fbd90b74ed43521b10f", 
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # === LLM 总结建议 ===
        try:
            history_wrong_focuses = []
            for exam in history.values():
                focus = exam.get("wrong_focus")
                if isinstance(focus, list):
                    history_wrong_focuses.extend(focus)
                elif isinstance(focus, str) and focus.strip():
                    history_wrong_focuses.append(focus)
            history_focus_text = "，".join(map(str, history_wrong_focuses)) if history_wrong_focuses else "无"
            current_wrong_focus_text = "，".join([f"{qid}-{focus}" for qid, focus in wrong_items]) if wrong_items else "无"

            prompt = f"""
    你是一个中学生学业辅导专家，请根据以下信息，为该学生撰写一段不超过150字的考试反馈建议，指出其进步与不足，并给予鼓励和改进建议。

    学生姓名：{info['name']}
    过往分数：{list(scores[:-1])}
    当前分数：{current_score}
    过往排名：{list(ranks[:-1])}
    当前排名：{current_rank}
    历史错误知识点：{history_focus_text}
    本次错误题目与知识点：{current_wrong_focus_text}
    请输出简洁凝练的反馈建议：
    """

            response = client.chat.completions.create(
                model='qwen3-32b', 
                messages=[{'role': 'user', 'content': prompt}],
                extra_body = {"enable_thinking": False}
            )
            advice_text = response.choices[0].message.content
            md_lines += ["\n## 💬 学习建议（由 AI 生成）\n", f"{advice_text}\n"]

        except Exception as e:
            print(f"⚠️ AI 生成建议失败：{e}")

        with open(os.path.join(student_dir, "student_report.md"), "w", encoding="utf-8") as f:
            f.writelines(md_lines)

        print(f"✅ 已生成报告：{student_dir}/student_report.md")

    # ========== 教师报告 ==========
    def parse_all_student_data(self,students_root="students"):
        all_student_records = []
        for sid in os.listdir(students_root):
            spath = os.path.join(students_root, sid)
            if not os.path.isdir(spath):
                continue
            try:
                with open(os.path.join(spath, "student_info.json"), "r", encoding="utf-8") as f:
                    info = json.load(f)
                with open(os.path.join(spath, "answer_revise.json"), "r", encoding="utf-8") as f:
                    ans = json.load(f)
                with open(os.path.join(spath, "exam_record.json"), "r", encoding="utf-8") as f:
                    hist = json.load(f)

                current_key = f"exam_{self.current_exam_number}"
                current_record = hist.get(current_key)
                current_rank = current_record["rank"]
                current_score = current_record["score"]


                all_student_records.append({
                    "student_id": info["student_id"],
                    "name": info["name"],
                    "gender": info["gender"],
                    "grade": self.grade,
                    "class": self.class_id,
                    "paper_id": self.paper_id,
                    "exam_number": self.current_exam_number,
                    "score": current_score,
                    "rank": current_rank,
                    "history": copy.deepcopy(hist),
                    "answers": ans,
                })

            except Exception as e:
                print(f"跳过 {sid}：{e}")
        return all_student_records

    def generate_teacher_report(self,all_student_records):
        os.makedirs("teacher_figures", exist_ok=True)

        # 一、分数分布
        scores = [s["score"] for s in all_student_records]
        plt.figure(figsize=(8, 4))
        sns.histplot(scores, bins=10, kde=True)
        plt.xlabel("分数")
        plt.ylabel("学生数量")
        plt.title("学生分数频率分布")
        plt.savefig("teacher_figures/score_distribution_histogram.png")
        plt.close()

        # 二、分数趋势
        exam_stats = defaultdict(lambda: {"score": []})
        for stu in all_student_records:
            for k, v in stu["history"].items():
                exam_num = int(k.split("_")[-1])
                exam_stats[exam_num]["score"].append(v["score"])

        stats_list = []
        for exam_num in sorted(exam_stats.keys()):
            scores = exam_stats[exam_num]["score"]
            stats_list.append({
                "exam": f"Exam {exam_num}",
                "mean": sum(scores) / len(scores),
                "max": max(scores),
                "min": min(scores),
                "median": median(scores)
            })

        df_stats = pd.DataFrame(stats_list)
        plt.figure(figsize=(8, 4))
        for col in ["mean", "max", "min", "median"]:
            plt.plot(df_stats["exam"], df_stats[col], marker="o", label=col)
        plt.ylabel("分数")
        plt.title("考试成绩统计趋势")
        plt.legend()
        plt.savefig("teacher_figures/score_statistics_trend.png")
        plt.close()

        # 三、题目 & 知识点 正确率（仅看是否满分）
        question_correct = defaultdict(list)
        knowledge_correct = defaultdict(lambda: {"correct": 0, "total": 0})

        for stu in all_student_records:
            for qid, ans in stu["answers"].items():
                if not isinstance(ans, dict):
                    continue
                is_correct = ans.get("get_full_point", False)
                question_correct[qid].append(is_correct)

                focus = ans.get("question_focus", "未知知识点")
                knowledge_correct[focus]["total"] += 1
                if is_correct:
                    knowledge_correct[focus]["correct"] += 1

        # 题目正确率图表
        question_accuracy_df = pd.DataFrame([{
            "question": qid,
            "accuracy": sum(v) / len(v)
        } for qid, v in question_correct.items()])
        question_accuracy_df = question_accuracy_df.sort_values("question", key=lambda x: x.astype(int))

        min_5 = question_accuracy_df.nsmallest(5, "accuracy")["question"].tolist()
        bar_colors = ["red" if q in min_5 else "blue" for q in question_accuracy_df["question"]]

        plt.figure(figsize=(10, 4))
        sns.barplot(x="question", y="accuracy", data=question_accuracy_df, palette=bar_colors)
        plt.title("每道题正确率")
        plt.savefig("teacher_figures/question_accuracy_bar.png")
        plt.close()

        # 知识点正确率图表
        knowledge_df = pd.DataFrame([{
            "knowledge": k,
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0
        } for k, v in knowledge_correct.items()])
        knowledge_df = knowledge_df.sort_values("accuracy")

        plt.figure(figsize=(10, 4))
        sns.barplot(x="knowledge", y="accuracy", data=knowledge_df, palette="Blues_d")
        plt.xticks(rotation=45, ha="right")
        plt.title("知识点正确率")
        plt.tight_layout()
        plt.savefig("teacher_figures/knowledge_accuracy_bar.png")
        plt.close()

        # 四、前五名 / 进步 / 退步
        top5 = sorted(all_student_records, key=lambda x: x["rank"])[:5]
        top5_names = [s["name"] for s in top5]

        for stu in all_student_records:
            history = stu["history"]

            # 获取所有 exam_x 中的 exam_number 和 rank（包括 exam_5）
            ranks = sorted(
                [(int(k.split("_")[-1]), v["rank"]) for k, v in history.items()]
            )

            # 分析排名变化：rank 越小越好（名次上升）
            stu["rank_diff"] = ranks[0][1] - ranks[-1][1] if len(ranks) >= 2 else 0
            stu["rank_range"] = f"{ranks[0][1]} → {ranks[-1][1]}" if len(ranks) >= 2 else "无"

        df = pd.DataFrame(all_student_records)
        progress_top5 = df.sort_values("rank_diff", ascending=False).head()
        decline_top5 = df.sort_values("rank_diff").head()

        # 五、平均排名表
        exam_range = range(1, 6)
        avg_rank_df = pd.DataFrame(index=[s["name"] for s in all_student_records])
        for e in exam_range:
            avg_rank_df[f"Exam {e}"] = [
                s["history"].get(f"exam_{e}", {}).get("rank", None) if s["exam_number"] != e else s["rank"]
                for s in all_student_records
            ]
        avg_rank_df["平均排名"] = avg_rank_df.mean(axis=1)
        avg_rank_df_sorted = avg_rank_df.sort_values("平均排名")
        avg_rank_df_sorted.to_excel("teacher_figures/五次考试平均排名表.xlsx")

        # 六、Markdown 报告写入
        today = datetime.date.today().strftime("%Y-%m-%d")
        md = f"""# 📘 教师报告汇总（{today}）

    ## 年级班级信息
    - 年级：{self.grade}  班级：{self.class_id}
    - 考试编号：{self.paper_id}  考试次数：Exam {self.current_exam_number}

    ## 一、学生分数频率分布直方图
    ![分数分布直方图](/teacher_figures/score_distribution_histogram.png)

    ## 二、分数统计变化趋势（五次考试）
    包括平均分、最高分、最低分、中位数随考试次数的变化：
    ![统计趋势图](/teacher_figures/score_statistics_trend.png)

    ## 三、每道题的正确率
    得分率最低的五题已用红色柱标出：
    ![题目得分率](/teacher_figures/question_accuracy_bar.png)

    ## 四、每个知识点的正确率
    ![知识点得分率](/teacher_figures/knowledge_accuracy_bar.png)

    ## 五、考试表现总结

    ### 🏆 最近一次考试前五名：
    {", ".join(top5_names)}

    ### 📈 进步前五名：
    """ + "\n".join([f"- {idx+1}. {row['name']}：↑{row['rank_diff']}名（{row['rank_range']}）" for idx, row in progress_top5.iterrows()]) + """

    ### 📉 退步前五名：
    """ + "\n".join([f"- {idx+1}. {row['name']}：↓{-row['rank_diff']}名（{row['rank_range']}）" for idx, row in decline_top5.iterrows()]) + """

    ## 六、五次考试平均排名表
    - 文件：`teacher_figures/五次考试平均排名表.xlsx`
    """

        # 七、AI 教学建议
        from openai import OpenAI
        client = OpenAI(
            api_key = "sk-10acd30d8f8c4fbd90b74ed43521b10f", 
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        try:
            worst_q = question_accuracy_df.nsmallest(5, "accuracy").values.tolist()
            worst_focus = knowledge_df.nsmallest(5, "accuracy").values.tolist()

            prompt = f"""你是一位中学教师，请根据以下数据撰写教学建议：

    - 正确率最低的题目：{', '.join([f"第{int(q[0])}题（{q[1]*100:.1f}%）" for q in worst_q])}
    - 正确率最低的知识点：{', '.join([f"{k[0]}（{k[1]*100:.1f}%）" for k in worst_focus])}
    - 进步前五：{', '.join([f"{s['name']}（+{s['rank_diff']}名）" for s in progress_top5.to_dict(orient='records')])}
    - 退步前五：{', '.join([f"{s['name']}（{s['rank_diff']}名）" for s in decline_top5.to_dict(orient='records')])}

    请：
    1. 指出重点讲解的题目/知识点；
    2. 针对学生表现提出建议；
    3. 给进步学生鼓励；
    4. 给退步学生提供关注建议。
    """



            response = client.chat.completions.create(
                model='qwen3-32b',  
                messages=[{'role': 'user', 'content': prompt}],
                extra_body = {"enable_thinking": False}
            )
            advice_text = response.choices[0].message.content
            md += "\n## 七、💡 AI 教学建议\n" + advice_text + "\n"
        except Exception as e:
            print(f"⚠️ 教学建议生成失败：{e}")
            md += "\n## 七、💡 AI 教学建议\n生成失败。\n"

        with open("teacher_report.md", "w", encoding="utf-8") as f:
            f.write(md)

        print("✅ 教师报告生成成功！")

# ========== 使用示例 ==========
# if __name__ == "__main__":
#     manager = EduAgent()
#     print("🤖正在提取试卷答案")
#     manager.student_s_answers_extractor(student_id=2024001)
#     print("✅成功提取第一个学生的试卷答案")
#     manager.scoring_executor(student_id=2024001)
#     print("✅成功批改第一个学生的试卷答案")
#     student_id_to_rank_dic = manager.generate_student_exam_rank()
#     print("✅成功产生本次考试的排名")
#     manager.write_student_exam_result_into_exam_record(student_id_to_rank_dic)
#     print("✅录入本次考试每个学生的考试记录")
#     # 批量生成学生报告
#     for sid in os.listdir(manager.students_root):
#         student_path = os.path.join(manager.students_root, sid)
#         if os.path.isdir(student_path):
#             try:
#                 manager.generate_report(student_path)
#             except Exception as e:
#                 print(f"⚠️ 跳过 {sid}：{e}")
#     # 生成教师报告
#     records = manager.parse_all_student_data()
#     manager.generate_teacher_report(records)
