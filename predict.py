"""
FilePath: /crack-0921/predict.py
Description:  
Author: rthete
Date: 2023-11-09 21:30:11
LastEditTime: 2023-11-11 17:30:10
"""
import cv2
from PIL import Image
from predict_kit.Processor import Processor


if __name__ == "__main__":
    
    test_img_path = "data/measure_data/measure_test_block_original_14_1 - 1.1mm - HK-HJD-S-01_181210_K2055+516.17_T2122.17_0595-01-00_VL-Y.jpg"
    # test_img_path = "data/measure_data/measure_test_block_original_1 - 1.1mm - HK-HJD-S-01_181210_K2055+516.17_T2122.17_0595-01-00_VL-Y.jpg"
    original_block = Image.open(test_img_path)
    segmented_img = Processor.segment_UNet(original_block)
    calibrated_img = Processor.calibrate(original_block, segmented_img)
    measured_img = Processor.measure_incircle(original_block, calibrated_img)
    cv2.imwrite("measured_img.jpg", measured_img)
    cv2.waitKey(0)