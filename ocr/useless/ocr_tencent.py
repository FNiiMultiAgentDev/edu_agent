import cv2
import pytesseract
from PIL import Image

# 如果没有设置环境变量，请取消下一行注释并指定路径
pytesseract.pytesseract.tesseract_cmd = r'F:\Tesseract-OCR\tesseract.exe'

# 读取图像
image_path = 'input1.png'  # 替换为你的图片路径
image = cv2.imread(image_path)
h, w, _ = image.shape

# 使用pytesseract逐字识别
boxes = pytesseract.image_to_boxes(image, lang='chi_sim')  # 中文可加 'chi_sim'，英文可用 'eng'

# 遍历每一个字符
for box in boxes.splitlines():
    char, x1, y1, x2, y2, _ = box.split()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 注意：Tesseract的Y轴是从图像底部开始的，因此需要转换
    y1 = h - y1
    y2 = h - y2

    print(f"字符：{char}，坐标：(left={x1}, top={y2}, right={x2}, bottom={y1})")

    # 可视化（可选）
    cv2.rectangle(image, (x1, y2), (x2, y1), (0, 255, 0), 2)
    cv2.putText(image, char, (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

# 显示结果图像（可选）
cv2.imshow("Character Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
