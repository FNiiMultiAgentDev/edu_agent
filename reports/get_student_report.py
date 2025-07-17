import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def generate_report(student_dir):
    # 预设的全局信息
    fixed_grade = "高三"
    fixed_class = "5班"
    fixed_paper_id = "第二学期 第五次考试"
    current_exam_number = 5  # 本次考试是 exam_5

    # 读取 json 文件
    with open(os.path.join(student_dir, "answer_revise.json"), "r", encoding="utf-8") as f:
        answers = json.load(f)
    with open(os.path.join(student_dir, "student_info.json"), "r", encoding="utf-8") as f:
        info = json.load(f)
    with open(os.path.join(student_dir, "exam_record.json"), "r", encoding="utf-8") as f:
        history = json.load(f)

    # 当前考试信息
    current_exam = current_exam_number
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
        f"- 班级：{fixed_grade} {fixed_class}\n",
        f"- 性别：{info['gender']}\n",
        f"- 考试编号：{fixed_paper_id}\n",
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
            model='qwen3-32b',  # 或者 qwen-plus, qwen-max 等
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

# ================== 批量运行 ==================
root_path = "students"
for sid in os.listdir(root_path):
    student_path = os.path.join(root_path, sid)
    if os.path.isdir(student_path):
        try:
            generate_report(student_path)
        except Exception as e:
            print(f"⚠️ 跳过 {sid}：{e}")
