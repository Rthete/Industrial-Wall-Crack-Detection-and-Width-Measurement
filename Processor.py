import copy
import cv2
import logging
import numpy as np
import string
import sys

from datetime import datetime
from PIL import Image

sys.path.append("..")

from model import Model
from utils import cvtColor

current_time = datetime.now().strftime("%Y%m%d%H%M%S")

logging.basicConfig(
    filename=f"logs/log_{current_time}.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Processor:
    @staticmethod
    def segment_UNet(original_img):
        """Stage 1. Use pretrained UNet model to predict crack areas."""

        count = False
        resnet50_unet = Model(model_path='pth/resnet50_unet_best_epoch_weights.pth', backbone="resnet50")
        vgg_unet = Model(model_path='pth/vgg_unet_best_epoch_weights.pth', backbone="vgg")
        name_classes = ["background", "crack"]

        pr_resnet50 = resnet50_unet.get_pr(
            original_img, count=count, name_classes=name_classes
        )
        pr_vgg = vgg_unet.get_pr(original_img, count=count, name_classes=name_classes)

        pr_resnet50_flat = pr_resnet50.flatten()
        unique_classes_resnet50, class_counts_resnet50 = np.unique(
            pr_resnet50_flat, return_counts=True
        )
        num_classes_resnet50 = len(unique_classes_resnet50)
        
        pr_vgg_flat = pr_vgg.flatten()
        unique_classes_vgg, class_counts_vgg = np.unique(
            pr_vgg_flat, return_counts=True
        )
        num_classes_vgg = len(unique_classes_vgg)

        tag_resnet50 = False
        tag_vgg = False
        pr_threshold = 1000
        if num_classes_resnet50 > 1 and class_counts_resnet50[1] > pr_threshold:
            tag_resnet50 = True
        if num_classes_vgg > 1 and class_counts_vgg[1] > pr_threshold:
            tag_vgg = True
        image = cvtColor(original_img)

        colors = [
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
            (128, 64, 12),
        ]
        positive_flag = False
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        if tag_resnet50 and tag_vgg:
            positive_flag = True
            segmented_img = np.reshape(
                np.array(colors, np.uint8)[np.reshape(pr_vgg, [-1])],
                [orininal_h, orininal_w, -1],
            )
        else:
            segmented_img = np.zeros((orininal_h, orininal_w))

        return segmented_img, positive_flag

    @staticmethod
    def edge_detection(original_img, segmented_img):
        """Stage 2. Edge detection to further locate crack areas."""

        original_img = np.array(original_img)
        try:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            pass
            # logging.error(f"Exception in edge detection: {e}")

        original_img = cv2.convertScaleAbs(original_img, alpha=0.4, beta=50)
        original_img = cv2.bitwise_not(original_img)

        edges = cv2.Canny(original_img, 50, 100)
        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)
        segmented_img = np.uint8(segmented_img > 0)
        result_img = cv2.bitwise_and(closed_edges, closed_edges, mask=segmented_img)

        return result_img

    @staticmethod
    def measure_incircle(index, original_img, edge_detected_img):
        """Stage 3. Incircle algorithm to measure segmented crack areas' width."""
        
        original_img = np.array(original_img)
        contours_arr, _ = cv2.findContours(
            edge_detected_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        raw_dist = np.zeros(edge_detected_img.shape, dtype=np.float32)
        label = []

        for _, contours in enumerate(contours_arr):
            for i in range(edge_detected_img.shape[0]):
                for j in range(edge_detected_img.shape[1]):
                    raw_dist[i, j] = cv2.pointPolygonTest(contours, (j, i), True)
            _, max_val, _, max_dist_pt = cv2.minMaxLoc(raw_dist)

            label.append(max_val * 2)
            radius = int(max_val)
            
            cv2.circle(original_img, max_dist_pt, radius, (0, 0, 255), 1, 1, 0)
            cv2.putText(
                original_img,
                f"{label[-1]:.2f}",
                (max_dist_pt[0] + radius + 5, max_dist_pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        
        logging.info(f"Widths in block {index}: {label}")

        return original_img

    @staticmethod
    def add_mask(segmented_img_array, old_img_array):
        """Stage 3. Add mask for result visualization."""
        
        segmented_img_Image = Image.fromarray(segmented_img_array)
        mask = np.all(segmented_img_Image == [128, 0, 0], axis=-1)

        alpha = 0.7
        mixed_pixels = (1 - alpha) * old_img_array[mask] + alpha * segmented_img_Image[mask]
        old_img_array[mask] = mixed_pixels
        old_img_Image = Image.fromarray(old_img_array)
        
        blend_factor = 0.5
        result_Image = Image.blend(old_img_Image, segmented_img_Image, blend_factor)
        
        return np.array(result_Image)

    @staticmethod
    def split_image(img_path, block_size):
        image = Image.open(img_path)
        img_width, img_height = image.size
        blocks = []
        logging.info(f"Splitting original image...original image size: `({str(img_height)}, {str(img_width)})`")
        for y in range(0, img_height, block_size):
            for x in range(0, img_width, block_size):
                box = (x, y, x + block_size, y + block_size)
                blocks.append(image.crop(box))
        logging.info(f"Original image is splitted into `{str(len(blocks))}` blocks.")
        return blocks

    @staticmethod
    def join_image(blocks, img_width, img_height):
        result_img = Image.new("RGB", (img_width, img_height))
        x, y = 0, 0
        for block in blocks:
            block = Image.fromarray(block)
            result_img.paste(block, (x, y))
            x += block.size[0]
            if x >= img_width:
                x = 0
                y += block.size[1]
        return result_img
