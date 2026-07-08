from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayoutV2")
output = model.predict(
    "path/to/your/page.jpg",
    batch_size=1,
    layout_nms=True  # 是否做布局 NMS 合并
)

for res in output:
    res.print()           # 打印每个检测框的信息（坐标、类别、score）
    res.save_to_img("out")   # 画在图上保存
    res.save_to_json("out")  # 保存为 JSON，里面有 bbox + label + order 等