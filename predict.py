import cv2
import numpy as np
import os
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed
from Processor import Processor
from utils.write2XML import write2XML


Image.MAX_IMAGE_PIXELS = None

write_xml_flag = True

def process_pipeline(index, original_block):
    segmented_block, positive_flag = Processor.segment_UNet(original_block)
    if not positive_flag:
        return np.array(original_block), None
    edge_detected_block, positive_flag = Processor.edge_detection(original_block, segmented_block)
    if not positive_flag:
        return np.array(original_block), None
    if write_xml_flag:
        return None, edge_detected_block
    measured_block = Processor.measure_incircle(
        index, original_block, edge_detected_block
    )
    result_block = Processor.add_mask(edge_detected_block, measured_block)
    return result_block, edge_detected_block


if __name__ == "__main__":
    sample_path = "data/"
    image_path_list = []
    for root, dirs, files in os.walk(sample_path):
        for file in files:
            image_path_list.append(os.path.join(root, file))

    for image_path in image_path_list:
        original_image = Image.open(image_path)
        blocks, locations = Processor.split_image(image_path, 1024)
        processed_blocks = []
        crack_areas = []
        crack_locations = []
        index = 0
        for block, location in zip(blocks, locations):
            index = index + 1
            processed_block, crack_area_np = process_pipeline(index, block)
            processed_blocks.append(processed_block)
            if crack_area_np is not None:
                crack_areas.append(crack_area_np)
                crack_locations.append(location)

        processed_image = Processor.join_image(
            processed_blocks, original_image.width, original_image.height
        )
        processed_image.save(
            "output/reg_img/{}".format(os.path.basename(image_path))
        )
        if write_xml_flag:
            write2XML(crack_areas, crack_locations, os.path.basename(image_path))
