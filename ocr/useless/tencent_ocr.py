from PIL import Image, ImageDraw
from tencentcloud.common.credential import Credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ocr.v20181119 import ocr_client, models
import base64
import difflib
import json

# === 配置区域 ===
SECRET_ID   = "AKIDkNp6C5mGWt31lIbifU8gGl26yeT5UfcM"   # 替换为你的 SecretId
SECRET_KEY  = "aw50iCWeaJpH5tVvtnTwBDoSyPgkcKqo"      # 替换为你的 SecretKey
REGION      = "ap-guangzhou"                         # 根据实际情况选择区域

INPUT_PATH     = "input.png"     # 本地图片路径
OCR_JSON_PATH  = "ocr_result.json"  # 输出 OCR 原始结果
OUTPUT_PATH    = "output.png"

# 官方原文库示例，可替换为你的完整库
official_texts = [
    "经济全球化",
    # …添加更多条目
]

# === 初始化 OCR 客户端 ===
cred = Credential(SECRET_ID, SECRET_KEY)
http_profile = HttpProfile(reqMethod="POST", reqTimeout=30)
client_profile = ClientProfile(httpProfile=http_profile)
client = ocr_client.OcrClient(cred, REGION, client_profile)

# === 读取本地图片并转 Base64 ===
with open(INPUT_PATH, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode()

# === 调用“试题识别”接口（QuestionOCR） ===
req = models.QuestionOCRRequest()
req.ImageBase64 = b64_data
resp = client.QuestionOCR(req)

# === 输出原始 OCR 结果到文件 ===
with open(OCR_JSON_PATH, "w", encoding="utf-8") as f:
    f.write(resp.to_json_string(indent=2, ensure_ascii=False))
print(f"OCR 结果已保存到：{OCR_JSON_PATH}")

# === 打开本地图片用于可视化标注 ===
image = Image.open(INPUT_PATH)
draw = ImageDraw.Draw(image, 'RGBA')

# === 遍历识别结果并做 difflib 相似度检测、标注 ===
for info in resp.QuestionInfo:
    for res in info.ResultList:
        for q in res.Question:
            text = q.Text
            coord = q.Coord
            bbox = [
                (coord.LeftTop.X,    coord.LeftTop.Y),
                (coord.RightTop.X,   coord.RightTop.Y),
                (coord.RightBottom.X,coord.RightBottom.Y),
                (coord.LeftBottom.X, coord.LeftBottom.Y),
            ]

            # difflib 计算相似度
            best_ratio = 0.0
            for orig in official_texts:
                ratio = difflib.SequenceMatcher(None, text, orig).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio

            # 相似度超过阈值则标注
            if best_ratio > 0.9:
                draw.polygon(bbox, fill=(255, 0, 0, 100))

# === 保存标注后的图片 ===
image.save(OUTPUT_PATH)
print(f"标注完成，结果已保存到：{OUTPUT_PATH}")
