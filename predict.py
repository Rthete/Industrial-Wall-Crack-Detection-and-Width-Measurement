import cv2
from datetime import datetime
import logging
import numpy as np
import os
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed
from Processor import Processor
from utils.write2XML import write2XML
from utils.datasetFilter import get_file_names


Image.MAX_IMAGE_PIXELS = None

write_xml_flag = True
write_npy_flag = False

current_time = datetime.now().strftime("%Y%m%d%H%M%S")
logging.basicConfig(
    filename=f"logs/log_{current_time}.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def process_pipeline(index, original_block):
    segmented_block, positive_flag = Processor.segment_UNet(original_block)
    if not positive_flag:
        return np.array(original_block), None
    if write_xml_flag:
        result_block = Processor.add_mask(np.uint8(segmented_block), np.array(original_block))
        if len(segmented_block.shape) == 3:
            segmented_block = cv2.cvtColor(segmented_block, cv2.COLOR_RGB2GRAY)
        segmented_block = np.uint8(segmented_block > 0)
        return result_block, segmented_block
    edge_detected_block, positive_flag = Processor.edge_detection(original_block, segmented_block)
    if not positive_flag:
        return np.array(original_block), None
    
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
    
    done_files = get_file_names("output/reg_img")
    
    for image_path in image_path_list:
        # 检查是否已经推理过
        if os.path.splitext(os.path.basename(image_path))[0] in done_files:
            logging.warning(f"done file `{os.path.splitext(image_path)[0]}`, skip")
            continue
        logging.info(f"processing: {image_path}")
        original_image = Image.open(image_path)
        blocks, locations = Processor.split_image(image_path, 1024)
        processed_blocks = []
        crack_areas = []
        crack_locations = []
        crack_np_list = []
        index = 0
        for block, location in zip(blocks, locations):
            index = index + 1
            processed_block, crack_area_np = process_pipeline(index, block)
            processed_blocks.append(processed_block)
            if crack_area_np is not None:
                crack_np_list.append(crack_area_np)
                crack_areas.append(crack_area_np)
                crack_locations.append(location)
            else:
                crack_np_list.append(np.zeros((1024, 1024)))

        processed_image = Processor.join_image(
            processed_blocks, original_image.width, original_image.height
        )
        processed_image.save(
            "output/reg_img/{}".format(os.path.basename(image_path))
        )
        
        if write_npy_flag:
            result_numpy = np.array(Processor.join_image(crack_np_list, original_image.width, original_image.height))
            np.save("output/reg_npy/{}.npy".format(os.path.basename(image_path)), result_numpy)
        if write_xml_flag:
            write2XML(crack_areas, crack_locations, os.path.basename(image_path))
