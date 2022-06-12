"""
图像特征点的检测与匹配
主要涉及：
1、ORB
2、SIFT
3、SURF
"""

"""
一、图像特征点的检测
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt


def ORB(img):
    """
     ORB角点检测
     实例化ORB对象
    """
    orb = cv2.ORB_create(nfeatures=500)
    """检测关键点，计算特征描述符"""
    kp, des = orb.detectAndCompute(img, None)

    # 将关键点绘制在图像上
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    # 画图
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img2[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


def SIFT(img):
    # SIFT算法关键点检测
    # 读取图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SIFT关键点检测
    # 1. 实例化sift
    sift = cv2.xfeatures2d.SIFT_create()

    # 2. 利用sift.detectAndCompute()检测关键点并计算
    kp, des = sift.detectAndCompute(gray, None)
    # gray: 进行关键带你检测的图像，注意是灰度图像
    # kp: 关键点信息，包括位置，尺度，方向信息
    # des: 关键点描述符，每个关键点对应128个梯度信息的特征向量

    # 3. 将关键点检测结果绘制在图像上
    # cv2.drawKeypoints(image, keypoints, outputimage, color, flags)
    # image: 原始图像
    # keypoints: 关键点信息，将其绘制在图像上
    # outputimage: 输出图片，可以是原始图像
    # color: 颜色设置，通过修改(b, g, r)的值，更改画笔的颜色，b = 蓝色, g = 绿色, r = 红色
    # flags: 绘图功能的标识设置
    # 1. cv2.DRAW_MATCHES_FLAGS_DEFAULT: 创建输出图像矩阵，使用现存的输出图像绘制匹配对象和特征点，对每一个关键点只绘制中间点
    # 2. cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG: 不创建输出图像矩阵，而是在输出图像上绘制匹配对
    # 3. cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 对每一个特征点绘制带大小和方向的关键点图形
    # 4. cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS: 单点的特征点不被绘制
    cv2.drawKeypoints(img, kp, img, (0, 255, 0))

    # 图像显示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


def SURF(img):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    cv2.drawKeypoints(img, kp, img, (0, 255, 0))

    # 图像显示
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


"""
2.图像特征点匹配方法
（1）暴力法
（2）FLANN匹配器
"""


def ByBFMatcher(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
    """
    （1）暴力法
    :param img1: 匹配图像1
    :param img2: 匹配图像2
    :param kp1: 匹配图像1的特征点
    :param kp2: 匹配图像2的特征点
    :param des1: 匹配图像1的描述子
    :param des2: 匹配图像2的描述子
    :return:
    """
    if (flag == "SIFT" or flag == "sift"):
        # SIFT方法
        bf = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
    else:
        # ORB方法
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
    ms = bf.knnMatch(des1, des2, k=2)
    # ms = sorted(ms, key=lambda x: x.distance)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, ms, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Matches", img3)
    # cv2.waitKey(0)
    return ms


def ByFlann(img1, img2, kp1, kp2, des1, des2, flag="ORB"):
    """
        （1）FLANN匹配器
        :param img1: 匹配图像1
        :param img2: 匹配图像2
        :param kp1: 匹配图像1的特征点
        :param kp2: 匹配图像2的特征点
        :param des1: 匹配图像1的描述子
        :param des2: 匹配图像2的描述子
        :return:
        """
    if (flag == "SIFT" or flag == "sift"):
        # SIFT方法
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                            trees=5)
        search_params = dict(check=50)
    else:
        # ORB方法
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(check=50)
    # 定义FLANN参数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches


"""
优化匹配结果
RANSAC算法是RANdom SAmple Consensus的缩写,意为随机抽样一致
"""


def RANSAC(img1, img2, kp1, kp2, matches,count):
    MIN_MATCH_COUNT = 10
    # store all the good matches as per Lowe's ratio test.
    matchType = type(matches[0])
    good = []
    # print(matchType)
    if isinstance(matches[0], cv2.DMatch):
        # 搜索使用的是match
        good = matches
    else:
        # 搜索使用的是knnMatch
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         good.append(m)
        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < 0.7*n.distance:
                    good.append(m)
            except ValueError:
                pass

    
    # good保存的是正确的匹配
    # 单应性
    if len(good) > MIN_MATCH_COUNT:
        # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M: 3x3 变换矩阵.
        # findHomography 函数是计算变换矩阵
        # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
        # 返回值：M 为变换矩阵，mask是掩模
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # ravel方法将数据降维处理，最后并转换成列表格式
        matchesMask = mask.ravel().tolist()

        # h, w = img1.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        #
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # 检测 画框
        # 获取img1的图像尺寸
        h, w, dim = img1.shape
        # pts是图像img1的四个顶点
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # 计算变换后的四个顶点坐标位置
        dst = cv2.perspectiveTransform(pts, M)
    
        # 根据四个顶点坐标位置在img2图像画出变换后的边框
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        print
        "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    # 显示匹配结果
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # draw_params1 = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                     singlePointColor=None,
    #                     matchesMask=None,  # draw only inliers
    #                     flags=2)

    # img33 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params1)

    # cv2.imshow("before", img33)
    # cv2.imshow("now", img3)
    if len(good) > MIN_MATCH_COUNT:
        cv2.imwrite("match/RANSAC_match"+str(count)+".jpg", img3)
        cv2.waitKey(0)


# img1 = cv2.imread("G:\\fan\\2022\object_detection\data\images\image1\image16.jpg")
# img2 = cv2.imread("G:\\fan\\2022\object_detection\data\images\image1\image20.jpg")
# kp1, des1 = ORB(img1)
# kp2, des2 = ORB(img2)
# matches = ByFlann(img1, img2, kp1, kp2, des1, des2, "ORB")
# RANSAC(img1, img2, kp1, kp2, matches)

