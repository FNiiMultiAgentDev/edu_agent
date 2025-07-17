import socket

def send_png_to_printer(printer_ip, printer_port, png_path):
    with open(png_path, 'rb') as f:
        png_data = f.read()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((printer_ip, printer_port))
        s.sendall(png_data)

    print(f"✅ PNG 文件已发送到打印机 {printer_ip}:{printer_port}")

# 使用示例
send_png_to_printer(
    printer_ip="192.168.1.88",       # 你的打印机 IP
    printer_port=9100,
    png_path="email_demo/answer_explanation.png"   # 你的 PNG 文件路径
)

# from pdf2image import convert_from_path
# import os

# # 设置 PDF 路径和输出目录
# pdf_path = "email_demo/teacher_report.pdf"
# output_dir = "email_demo/pages"

# # 创建输出目录（如果不存在）
# os.makedirs(output_dir, exist_ok=True)

# # 转换 PDF 所有页面为图片
# images = convert_from_path(
#     pdf_path,
#     poppler_path=r"F:\poppler-24.08.0\Library\bin"  # 一定要写这一行
# )

# # 保存每一页为 PNG 格式
# for i, image in enumerate(images):
#     image_path = os.path.join(output_dir, f"page_{i+1}.png")
#     image.save(image_path, "PNG")
#     print(f"✅ 已保存：{image_path}")
