from PIL import Image, ImageDraw
best_match= [[90, 0], [1077, 3], [1077, 83], [90, 80],[0,90],[500,90],[500,150],[50,150]]

# 加载原图（确保是RGBA）
base = Image.open("ocr/input.png").convert("RGBA")

# 创建一个透明图层
overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

i = 0
while i < len(best_match):

    draw.polygon(best_match[i: i+4], fill=(255, 0, 0, 100))  # alpha=100 表示半透明
    i += 4


# 合成原图与染色图层
out = Image.alpha_composite(base, overlay)

# 保存最终效果
out.save("ocr/output_baidu.png")
print("✅ 染色结果已更新为半透明红色并保存为 output_baidu.png")