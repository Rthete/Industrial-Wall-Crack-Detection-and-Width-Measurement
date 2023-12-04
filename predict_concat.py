import os
import numpy as np
from PIL import Image
from unet import Unet
import copy
from utils.utils import cvtColor
from unet2 import Unet2

# True: 生成供宽度测量实验的裂缝区域二值图像
measure_flag = True

Image.MAX_IMAGE_PIXELS = None
count = False
name_classes = ["background","crack"]
resnet50_unet = Unet()
vgg_unet = Unet2()
colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
          (128, 64, 12)]


def process_block(block, file, index):
    pr_resnet50 = resnet50_unet.get_pr(block, count=count, name_classes=name_classes)
    pr_vgg = vgg_unet.get_pr(block, count=count, name_classes=name_classes)
    
    # 将预测的概率图（pr）转换为一维数组
    pr_resnet50_flat = pr_resnet50.flatten()
    
    # 获取唯一的像素种类和它们的出现次数
    unique_classes_resnet50, class_counts_resnet50 = np.unique(pr_resnet50_flat, return_counts=True)
    
    # 唯一像素种类的个数
    num_classes_resnet50 = len(unique_classes_resnet50)
    pr_vgg_flat = pr_vgg.flatten()
    unique_classes_vgg, class_counts_vgg = np.unique(pr_vgg_flat, return_counts=True)
    num_classes_vgg = len(unique_classes_vgg)

    # 判断是否取
    Tag_resnet50 = False
    Tag_vgg = False
    pr_threshold = 1000
    if num_classes_resnet50 > 1 and class_counts_resnet50[1] > pr_threshold:
        Tag_resnet50 = True
    if num_classes_vgg > 1 and class_counts_vgg[1] > pr_threshold:
        Tag_vgg = True
    image = cvtColor(block)
    
    if Tag_resnet50 and Tag_vgg:
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr_vgg, [-1])], [orininal_h, orininal_w, -1])

        # 将新图片转换成Image的形式
        image = Image.fromarray(np.uint8(seg_img))
        back_image = image
        
        # 保存一张block二值图以供宽度测量实验
        if measure_flag is True:
            image.save("data/measure_data/measure_test_block_{}_{}".format(index, file))
            old_img.save("data/measure_data/measure_test_block_original_{}_{}".format(index, file))
        
        # 将Image对象转换为NumPy数组
        old_img_array = np.array(old_img)
        image_array = np.array(image)

        # 创建掩膜，将image中RGB为(0,0,0)的像素位置置为True
        mask = np.all(image_array == [128, 0, 0], axis=-1)

        # 混合系数
        alpha = 0.7

        # 使用NumPy数组进行混合
        mixed_pixels = (1 - alpha) * old_img_array[mask] + alpha * image_array[mask]

        # 更新混合后的像素到old_img对应的位置
        old_img_array[mask] = mixed_pixels

        # 将NumPy数组转换为Image对象
        image = Image.fromarray(old_img_array)

        # 增加阴影效果
        # 创建与 image 相同大小的透明黑色图层
        blend_factor = 0.5
        image = Image.blend(image, back_image, blend_factor)

    return image


def split_image(image, block_size):
    img_width, img_height = image.size
    blocks = []
    print("原图片宽：",img_width)
    print("原图片高：",img_height)
    
    for y in range(0, img_height, block_size):
        for x in range(0, img_width, block_size):
            box = (x, y, x + block_size, y + block_size)
            blocks.append(image.crop(box))
    print("分为："+str(len(blocks))+'块')   
            
    return blocks


def join_image(blocks, img_width, img_height):
    result_img = Image.new("RGB", (img_width, img_height))
    x, y = 0, 0
    
    for block in blocks:
        result_img.paste(block, (x, y))
        x += block.size[0]
        if x >= img_width:
            x = 0
            y += block.size[1]
            
    return result_img


def get_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def open_or_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


if __name__ == "__main__":
    
    # 获取待分割的图片
    folder_path = 'data/seg_data'
    # folder_path = 'data/data_width'
    
    if measure_flag == True:
        # folder_path = 'data/measure_data'
        # folder_path = 'data/data_width'
        pass
    
    files = get_files_in_folder(folder_path)
    
    i = 1
    for file in files:
        # 读取原始图片
        file = os.path.basename(file)
        image_path = folder_path+'/'+file
        original_image = Image.open(image_path)

        # 定义分块大小
        block_size = 1024

        # 分割图像成小块
        blocks = split_image(original_image, block_size)

        # 对每个小块进行处理
        processed_blocks = []
        index = 0
        for block in blocks:
            index = index + 1
            processed_blocks.append(process_block(block, file, index))

        # 将各个block拼接回大图，并保存
        processed_image = join_image(processed_blocks, original_image.width, original_image.height)
        processed_image.save('output/result_{}'.format(file))
        
        print("已处理:"+str(i)+'/'+str(len(files))+'张')
        i = i + 1