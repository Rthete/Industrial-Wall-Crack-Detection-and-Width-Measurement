import cv2
import math
import random
import numpy as np
from numpy.ma import cos, sin


def max_circle(img, img_original, img_result):
    """
    获取裂缝区域最大内切圆

    Args:
        img: 分割算法获取的二值图像
        img_original: 待处理的原图像
        img_result: 结果图保存路径
    """

    img_original = cv2.imread(img_original)
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    
    # 灰度处理
    img_original = np.array(img_original)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bitwise_not(img_gray)
    
    #----------------------------------------------------------#
    # 图片二值化，缺少这一步也可以，但是图像的二值化可以使图
    # 像中数据量大为减少，从而能凸显出目标的轮廓，减小计算量
    #----------------------------------------------------------#
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    #----------------------------------------------------------#
    # 寻找二值图像的轮廓
    #
    # 第二个参数表示轮廓的检索模式，有四种：
    #     cv2.RETR_EXTERNAL表示只检测外轮廓
    #     cv2.RETR_LIST检测的轮廓不建立等级关系
    #     cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    #     cv2.RETR_TREE建立一个等级树结构的轮廓。
    # 第三个参数method为轮廓的近似办法
    #     cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
    #     cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    #     cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
    #----------------------------------------------------------#
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 所有裂缝最大内切圆半径和圆心列表
    expansion_circle_list = []
    
    # 可能一张图片中存在多条裂缝，对每一条裂缝进行循环计算
    for c in contours:
        
        # 定义能包含此裂缝的最小矩形，矩形为水平方向
        left_x = min(c[:, 0, 0])
        right_x = max(c[:, 0, 0])
        down_y = max(c[:, 0, 1])
        up_y = min(c[:, 0, 1])
        
        # 最小矩形中最小的边除2，裂缝内切圆的半径最大不超过此距离
        upper_r = min(right_x - left_x, down_y - up_y) / 2
        
        # 定义相切二分精度precision
        precision = math.sqrt((right_x - left_x) ** 2 + (down_y - up_y) ** 2) / (2 ** 13)
        
        # 构造包含轮廓的矩形的所有像素点
        Nx = 2 ** 8
        Ny = 2 ** 8
        pixel_X = np.linspace(left_x, right_x, Nx)
        pixel_Y = np.linspace(up_y, down_y, Ny)
        
        # 从坐标向量中生成网格点坐标矩阵
        xx, yy = np.meshgrid(pixel_X, pixel_Y)
        
        # 筛选出轮廓内所有像素点
        in_list = []
        for i in range(pixel_X.shape[0]):
            for j in range(pixel_X.shape[0]):
                ## cv2.pointPolygonTest可查找图像中的点与轮廓之间的最短距离.
                ## 当点在轮廓外时返回负值，当点在内部时返回正值，如果点在轮廓上则返回零
                # 统计裂缝内的所有点的坐标
                if cv2.pointPolygonTest(c, (xx[i][j], yy[i][j]), False) > 0:
                    in_list.append((xx[i][j], yy[i][j]))
        in_point = np.array(in_list)
        
        # 随机搜索百分之一的像素点提高内切圆半径下限
        N = len(in_point)
        rand_index = random.sample(range(N), N // 100)
        rand_index.sort()
        radius = 0
        
        # 裂缝内切圆的半径最大不超过此距离
        big_r = upper_r
        center = None
        for id in rand_index:
            tr = iterated_optimal_incircle_radius_get(c, in_point[id][0], in_point[id][1], radius, big_r, precision)
            # 只有半径变大才允许位置变更，否则保持之前位置不变
            if tr > radius:
                radius = tr
                center = (in_point[id][0], in_point[id][1])  
        
        # 循环搜索剩余像素对应内切圆半径
        loops_index = [i for i in range(N) if i not in rand_index]
        for id in loops_index:
            tr = iterated_optimal_incircle_radius_get(c, in_point[id][0], in_point[id][1], radius, big_r, precision)
             # 只有半径变大才允许位置变更，否则保持之前位置不变
            if tr > radius:
                radius = tr
                center = (in_point[id][0], in_point[id][1])

        # 保存每条裂缝最大内切圆的半径和圆心
        expansion_circle_list.append([radius, center])
           
        # 输出裂缝的最大宽度
        print('裂缝宽度：', round(radius * 2, 2))

    print('---------------')
    
    # 每条裂缝最大内切圆半径列表
    expansion_circle_radius_list = [i[0] for i in expansion_circle_list]   
    if len(expansion_circle_radius_list) > 0: 
        max_radius = max(expansion_circle_radius_list)
        max_center = expansion_circle_list[expansion_circle_radius_list.index(max_radius)][1]
        print('最大宽度：', round(max_radius * 2, 2))
        # 绘制轮廓
        cv2.drawContours(img_original, contours, -1, (0, 0, 255), -1)
        # 绘制裂缝轮廓最大内切圆
        for expansion_circle in expansion_circle_list:
            radius_s = expansion_circle[0]
            center_s = expansion_circle[1]
            if radius_s > 0:
                if radius_s == max_radius: 
                    # 最大内切圆，用蓝色标注 
                    cv2.circle(img_original, (int(max_center[0]), int(max_center[1])), int(max_radius), (255, 0, 0), 2)
                else:
                    # 其他内切圆，用青色标注
                    cv2.circle(img_original, (int(center_s[0]), int(center_s[1])), int(radius_s), (255, 245, 0), 2)
                formatted_width = f"{radius_s * 2:.2f}"
                cv2.putText(img_original, f"{formatted_width}", (int(center_s[0]-10), int(center_s[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.imwrite(img_result, img_original)


def iterated_optimal_incircle_radius_get(contours, pixelx, pixely, small_r, big_r, precision):
    '''
    计算轮廓内最大内切圆的半径
    
    Args:
        contours: 轮廓像素点array数组
        pixelx: 圆心x像素坐标
        pixely: 圆心y像素坐标
        small_r: 之前所有计算所求得的内切圆的最大半径，作为下次计算时的最小半径输入，只有半径变大时才允许位置变更，否则保持之前位置不变
        big_r: 圆的半径最大不超过此距离
        precision: 相切二分精度，采用二分法寻找最大半径

    Returns: 轮廓内切圆的半径
    '''
    radius = small_r
    # 确定圆散点剖分数360, 720
    L = np.linspace(0, 2 * math.pi, 360)  
    circle_X = pixelx + radius * cos(L)
    circle_Y = pixely + radius * sin(L)
    for i in range(len(circle_Y)):
        # 如果圆散集有在轮廓之外的点
        if cv2.pointPolygonTest(contours, (circle_X[i], circle_Y[i]), False) < 0:  
            return 0
    while big_r - small_r >= precision:  # 二分法寻找最大半径
        half_r = (small_r + big_r) / 2
        circle_X = pixelx + half_r * cos(L)
        circle_Y = pixely + half_r * sin(L)
        if_out = False
        for i in range(len(circle_Y)):
            # 如果圆散集有在轮廓之外的点
            if cv2.pointPolygonTest(contours, (circle_X[i], circle_Y[i]), False) < 0:
                big_r = half_r
                if_out = True
        if not if_out:
            small_r = half_r
    radius = small_r
    return radius


if __name__ == '__main__':
    # 灰度图
    img_gray = 'data/measure_data/8-0_3mm/measure_test_block_5_8 - 0.3mm - HK-HJD-S-01_181210_K2056+066.36_T2672.36_0749-01-00_VL-Y.jpg'
    # 原图
    img_original = 'data/measure_data/8-0_3mm/measure_test_block_original_5_8 - 0.3mm - HK-HJD-S-01_181210_K2056+066.36_T2672.36_0749-01-00_VL-Y.jpg'
    # 结果路径
    img_result = 'output/measure/result_measure_test_block_measure_test_block_original_5_8 - 0.3mm - HK-HJD-S-01_181210_K2056+066.36_T2672.36_0749-01-00_VL-Y.jpg'
    
    max_circle(img_gray, img_original, img_result) 

