import cv2
import numpy as np
import os
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed
from Processor import Processor


Image.MAX_IMAGE_PIXELS = None


def process_pipeline(index, original_block):
    segmented_block, positive_flag = Processor.segment_UNet(original_block)
    if not positive_flag:
        return np.array(original_block)
    edge_detected_block = Processor.edge_detection(original_block, segmented_block)
    measured_block = Processor.measure_incircle(
        index, original_block, edge_detected_block
    )
    result_block = Processor.add_mask(edge_detected_block, measured_block)
    return result_block
    # return measured_block


if __name__ == "__main__":
    sample_path = "data/"
    image_path_list = []
    for root, dirs, files in os.walk(sample_path):
        for file in files:
            image_path_list.append(os.path.join(root, file))

    for image_path in image_path_list:
        original_image = Image.open(image_path)
        blocks = Processor.split_image(image_path, 1024)
        processed_blocks = []
        index = 0
        for block in blocks:
            index = index + 1
            processed_blocks.append(process_pipeline(index, block))

        processed_image = Processor.join_image(
            processed_blocks, original_image.width, original_image.height
        )
        processed_image.save(
            "output/predict_result_{}".format(os.path.basename(image_path))
        )
