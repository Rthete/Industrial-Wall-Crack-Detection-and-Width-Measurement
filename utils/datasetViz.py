import os
from PIL import Image
import numpy as np

def overlay_images(original_folder, annotation_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取原图文件夹下所有文件
    original_files = os.listdir(original_folder)

    for file in original_files:
        # 构造原图和标注文件的路径
        original_path = os.path.join(original_folder, file)
        annotation_path = os.path.join(annotation_folder, file.replace('.jpg', '.png'))

        # 打开原图和标注图
        original_image = Image.open(original_path)
        annotation_image = Image.open(annotation_path)

        # 将标注图叠加在原图上
        result_image = Image.blend(original_image.convert('RGB'), annotation_image.convert('RGB'), alpha=0.5)

        # 保存结果图到输出文件夹
        result_path = os.path.join(output_folder, file.replace('.jpg', '_result.jpg'))
        result_image.save(result_path, 'JPEG')

        print(f"Saved: {result_path}")

if __name__ == "__main__":
    # 将标记文件与原图混合（供可视化）
    original_folder_path = "/mnt/d/Projects/project-0624/HJD-VOC/JPEGImages"
    annotation_folder_path = "/mnt/d/Projects/project-0624/HJD-VOC/SegmentationClassPNG"
    output_folder_path = "/mnt/d/Projects/project-0624/HJD-VOC/blend"

    overlay_images(original_folder_path, annotation_folder_path, output_folder_path)

