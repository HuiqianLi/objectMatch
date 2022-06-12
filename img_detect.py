import numpy as np
import cv2
from matplotlib import pyplot as plt

# ORB+单应性矩阵+特征匹配（Flann）+目标检测
# https://blog.csdn.net/qq_36071362/article/details/113617100

MIN_MATCH_COUNT = 10
 
img1 = cv2.imread('G:\\fan\\2022\object_detection\data\images\image1\image16.jpg', 1)
img2 = cv2.imread('G:\\fan\\2022\object_detection\data\images\image1\image20.jpg', 1)
 
exx = 0.8
exy = 0.8
 
img1 = cv2.resize(img1, dsize=None, fx=exx, fy=exy)
img2 = cv2.resize(img2, dsize=None, fx=exx, fy=exy)
 
'''img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)'''

# 使用ORB检测
orb = cv2.ORB_create(nfeatures=500)
"""检测关键点，计算特征描述符"""
# 获取关键点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print(type(kp1[1]))
# print(type(kp1))
print(type(des1))
# print(des1)

# kp是一个关键点列表
# des是一个numpy数组，其大小是关键点数目乘以128,描述符，描述关键点的位置，尺度，向量等信息
# kp = sift.detect(gray, None)
# kp, des = sift.compute(gray, kp)
 
# 在图中显示特征点
'''img_small_sift = cv2.drawKeypoints(img1, kp1, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_big_sift = cv2.drawKeypoints(img2, kp2, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("img1", img_small_sift)
cv2.imshow("img2", img_big_sift)'''
 
# 定义FLANN匹配器------KD树，具体原理尚不清楚，index_params中的algorithm为0或1
# index_params = dict(algorithm=1, trees=5)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=50)
# 定义FLANN参数
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 使用KNN算法匹配,FLANN里边就包含的KNN、KD树还有其他的最近邻算法
matches = flann.knnMatch(des1, des2, k=2)
# print(matches)
# 去除错误匹配
good = []
for m, n in matches: #m为第一邻近，n为第二邻近，两者距离满足以下关系才认为是正确匹配
    if m.distance <= 0.7 * n.distance:
        good.append(m)
 
# good保存的是正确的匹配
# 单应性
if len(good) > MIN_MATCH_COUNT:
    # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # findHomography 函数是计算变换矩阵
    # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
    # 返回值：M 为变换矩阵，mask是掩模
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # ravel方法将数据降维处理，最后并转换成列表格式
    matchesMask = mask.ravel().tolist()
    # 获取img1的图像尺寸
    h, w, dim = img1.shape
    # pts是图像img1的四个顶点
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # 计算变换后的四个顶点坐标位置
    dst = cv2.perspectiveTransform(pts, M)
 
    # 根据四个顶点坐标位置在img2图像画出变换后的边框
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
 
else:
    print("Not enough matches are found - %d/%d", (len(good), MIN_MATCH_COUNT))
    matchesMask = None
 
# 显示匹配结果
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
 
# 将画面显示
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.show()