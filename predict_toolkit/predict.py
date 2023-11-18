"""
FilePath: /crack-0921/predict_toolkit/predict.py
Description:  
Author: rthete
Date: 2023-11-09 21:30:11
LastEditTime: 2023-11-18 19:38:17
"""
import cv2
import os
from PIL import Image
from Processor import Processor
import numpy as np

Image.MAX_IMAGE_PIXELS = None

def process_pipeline(original_block):
    segmented_block, positive_flag = Processor.segment_UNet(original_block)
    if not positive_flag:
        return np.array(original_block)
    calibrated_block = Processor.calibrate(original_block, segmented_block)
    measured_block = Processor.measure_incircle(original_block, calibrated_block)
    return measured_block


if __name__ == "__main__":
    # sample_path = "D:\projects\project-0624\crack-0921\data\data_backup"
    sample_path = "D:\projects\project-0624\crack-0921\data\predict_toolkit_set"
    image_path_list = []
    for root, dirs, files in os.walk(sample_path):
        # print(files)
        for file in files:
            image_path_list.append(os.path.join(root, file))
    
    # print(image_path_list)
    # exit()
    for image_path in image_path_list:
        original_image = Image.open(image_path)
        blocks = Processor.split_image(image_path, 1024)
        processed_blocks = []
        index = 0
        for block in blocks:
            index = index + 1
            processed_blocks.append(process_pipeline(block))
            
        processed_image = Processor.join_image(processed_blocks, original_image.width, original_image.height)
        processed_image.save('result_{}'.format(os.path.basename(image_path)))

def test_processor():
    # test_img_path = "data/measure_data/measure_test_block_original_14_1 - 1.1mm - HK-HJD-S-01_181210_K2055+516.17_T2122.17_0595-01-00_VL-Y.jpg"
    test_img_path = "data/measure_data/measure_test_block_original_1 - 1.1mm - HK-HJD-S-01_181210_K2055+516.17_T2122.17_0595-01-00_VL-Y.jpg"
    original_block = Image.open(test_img_path)
    segmented_img = Processor.segment_UNet(original_block)
    calibrated_img = Processor.calibrate(original_block, segmented_img)
    measured_img = Processor.measure_incircle(original_block, calibrated_img)
    cv2.imwrite("measured_img.jpg", measured_img)
    cv2.waitKey(0)