import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from statistics import median
from collections import defaultdict
import datetime
import copy
# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 固定参数
FIXED_GRADE = "高三"
FIXED_CLASS = "5班"
FIXED_PAPER_ID = "第二学期 第五次考试"
CURRENT_EXAM_NUMBER = 5

# ============ 读取学生数据 ============

def parse_all_student_data(students_root="students"):
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

            current_key = f"exam_{CURRENT_EXAM_NUMBER}"
            current_record = hist.get(current_key)
            current_rank = current_record["rank"]
            current_score = current_record["score"]


            all_student_records.append({
                "student_id": info["student_id"],
                "name": info["name"],
                "gender": info["gender"],
                "grade": FIXED_GRADE,
                "class": FIXED_CLASS,
                "paper_id": FIXED_PAPER_ID,
                "exam_number": CURRENT_EXAM_NUMBER,
                "score": current_score,
                "rank": current_rank,
                "history": copy.deepcopy(hist),
                "answers": ans,
            })

        except Exception as e:
            print(f"跳过 {sid}：{e}")
    return all_student_records

# ============ 教师报告主函数 ============

def generate_teacher_report(all_student_records):
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
- 年级：{FIXED_GRADE}  班级：{FIXED_CLASS}
- 考试编号：{FIXED_PAPER_ID}  考试次数：Exam {CURRENT_EXAM_NUMBER}

## 一、学生分数频率分布直方图
![分数分布直方图](../teacher_figures/score_distribution_histogram.png)

## 二、分数统计变化趋势（五次考试）
包括平均分、最高分、最低分、中位数随考试次数的变化：
![统计趋势图](../teacher_figures/score_statistics_trend.png)

## 三、每道题的正确率
得分率最低的五题已用红色柱标出：
![题目得分率](../teacher_figures/question_accuracy_bar.png)

## 四、每个知识点的正确率
![知识点得分率](../teacher_figures/knowledge_accuracy_bar.png)

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

# ============ 执行 ============

records = parse_all_student_data("students")
generate_teacher_report(records)
