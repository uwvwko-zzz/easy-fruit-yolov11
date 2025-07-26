import torch
# 加载模型
from ultralytics import YOLO


model = YOLO(r'D:\yolov11\fruit_test\runs\detect\train\weights\best.pt')
# 进行推理
# 批量检测test文件夹中的所有图片并保存结果
results = model(r"D:\yolov11\fruit_test\fruit\test\images", save=True)

# 打印检测结果保存路径
for result in results:
    print(f"检测完成: {result.path}，结果保存至: {result.save_dir}")