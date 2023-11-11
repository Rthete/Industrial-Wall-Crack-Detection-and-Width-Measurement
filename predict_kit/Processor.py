import copy
import cv2
import numpy as np
from PIL import Image

from unet import Unet
from utils.utils import cvtColor
from unet2 import Unet2
from predict_kit.MaxCircle import MaxCircle

class Processor():
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

        if Tag_resnet50 and Tag_vgg:
            old_img = copy.deepcopy(image)
            orininal_h = np.array(image).shape[0]
            orininal_w = np.array(image).shape[1]
            seg_img = np.reshape(
                np.array(colors, np.uint8)[np.reshape(pr_vgg, [-1])],
                [orininal_h, orininal_w, -1],
            )

            # 将新图片转换成Image的形式
            # block_image = Image.fromarray(np.uint8(seg_img))
        return seg_img

    @staticmethod
    def calibrate(original_img, label_img):
        """Stage 2. Edge detection to further calibrate crack areas."""

        """Stage 2.1 原图预处理"""
        # 加载原图为灰度图
        original_img = np.array(original_img)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
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
        result_img = MaxCircle.max_circle(original_img, calibrated_img)
        return result_img


def process_block():
    """Predict each block respectively."""
    pass