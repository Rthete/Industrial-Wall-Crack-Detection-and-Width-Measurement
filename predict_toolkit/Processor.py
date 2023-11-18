import copy
import cv2
import numpy as np
import string
import sys
from PIL import Image
import logging

sys.path.append("..")

from unet import Unet
from unet2 import Unet2
from utils.utils import cvtColor

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Processor:
    @staticmethod
    def segment_UNet(original_block):
        """Stage 1. Use pretrained UNet model to predict crack areas."""

        count = False
        resnet50_unet = Unet()
        vgg_unet = Unet2()
        name_classes = ["background", "crack"]
        pr_resnet50 = resnet50_unet.get_pr(
            original_block, count=count, name_classes=name_classes
        )
        pr_vgg = vgg_unet.get_pr(original_block, count=count, name_classes=name_classes)

        # 将预测的概率图（pr）转换为一维数组
        pr_resnet50_flat = pr_resnet50.flatten()

        # 获取唯一的像素种类和它们的出现次数
        unique_classes_resnet50, class_counts_resnet50 = np.unique(
            pr_resnet50_flat, return_counts=True
        )

        # 唯一像素种类的个数
        num_classes_resnet50 = len(unique_classes_resnet50)
        pr_vgg_flat = pr_vgg.flatten()
        unique_classes_vgg, class_counts_vgg = np.unique(
            pr_vgg_flat, return_counts=True
        )
        num_classes_vgg = len(unique_classes_vgg)

        # 判断是否取
        Tag_resnet50 = False
        Tag_vgg = False
        pr_threshold = 1000
        if num_classes_resnet50 > 1 and class_counts_resnet50[1] > pr_threshold:
            Tag_resnet50 = True
        if num_classes_vgg > 1 and class_counts_vgg[1] > pr_threshold:
            Tag_vgg = True
        image = cvtColor(original_block)

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
        if Tag_resnet50 and Tag_vgg:
            positive_flag = True
            old_img = copy.deepcopy(image)
            seg_img = np.reshape(
                np.array(colors, np.uint8)[np.reshape(pr_vgg, [-1])],
                [orininal_h, orininal_w, -1],
            )
        else:
            seg_img = np.zeros((orininal_h, orininal_w))
            
        # # 将新图片转换成Image的形式
        # seg_img = np.array(seg_img)

        return seg_img, positive_flag

    @staticmethod
    def calibrate(original_img, label_img):
        """Stage 2. Edge detection to further calibrate crack areas."""

        """Stage 2.1 原图预处理"""
        # 加载原图为灰度图
        original_img = np.array(original_img)
        try:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            logging.error(f"clibrate: {e}")
        cv2.imwrite("predict_test/original_img.jpg", original_img)

        # 调整对比度
        original_img = cv2.convertScaleAbs(original_img, alpha=0.4, beta=50)
        original_img = cv2.bitwise_not(original_img)

        """Stage 2.2 获取边缘检测结果"""
        # canny边缘检测
        edges = cv2.Canny(original_img, 50, 100)
        # 定义闭运算的核
        kernel = np.ones((5, 5), np.uint8)
        # 闭运算将边缘检测结果填充
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("predict_test/CV2.jpg", closed_edges)

        """Stage 2.3 结合分割模型结果"""
        # 加载分割模型输出的二值图
        label_img = cv2.cvtColor(label_img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("predict_test/UNet.jpg", label_img)
        # 将label_img大于0的像素置为1，<=0的像素置为0
        label_img = np.uint8(label_img > 0)
        # 将分割模型的结果与边缘检测结果相与
        result_img = cv2.bitwise_and(closed_edges, closed_edges, mask=label_img)
        cv2.imwrite("predict_test/result_img.jpg", result_img)

        return result_img

    @staticmethod
    def measure_incircle(original_img, calibrated_img):
        """Stage 3. Incircle algorithm to measure segmented crack areas' width."""
        original_img = np.array(original_img)
        contours_arr, _ = cv2.findContours(
            calibrated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        # result：一个与输入图像相同大小的BGR图像，用于绘制测量结果。
        # raw_dist：一个与输入图像相同大小的浮点数数组，用于存储点到轮廓的距离。
        # letters：一个包含大写字母的列表，用于给测量的裂缝区域标签。
        # label：一个字典，用于存储每个裂缝区域的宽度测量结果。
        result = cv2.cvtColor(calibrated_img, cv2.COLOR_GRAY2BGR)
        raw_dist = np.zeros(calibrated_img.shape, dtype=np.float32)
        # letters = list(string.ascii_uppercase)
        # label = {}
        label = []

        # 遍历每个轮廓并测量宽度
        for k, contours in enumerate(contours_arr):
            for i in range(calibrated_img.shape[0]):
                for j in range(calibrated_img.shape[1]):
                    # 使用cv2.pointPolygonTest计算点到轮廓的距离
                    raw_dist[i, j] = cv2.pointPolygonTest(contours, (j, i), True)
            # 找到最大距离的点和对应的距离
            min_val, max_val, _, max_dist_pt = cv2.minMaxLoc(raw_dist)
            # 将裂缝宽度的两倍存储到label字典中
            # label[letters[k]] = max_val * 2
            label.append(max_val * 2)
            radius = int(max_val)
            # 在结果图像上绘制测量结果
            cv2.circle(original_img, max_dist_pt, radius, (0, 0, 255), 1, 1, 0)
        logging.info(f"裂缝宽度：{label}")

        return original_img


    @staticmethod
    def split_image(img_path, block_size):
        image = Image.open(img_path)
        img_width, img_height = image.size
        blocks = []
        logging.info(f"原图height: {str(img_height)}, 原图width: {str(img_width)}")
        for y in range(0, img_height, block_size):
            for x in range(0, img_width, block_size):
                box = (x, y, x + block_size, y + block_size)
                blocks.append(image.crop(box))
        logging.info(f"分为：{str(len(blocks))}块")   
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