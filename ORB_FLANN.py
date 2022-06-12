import cv2
import numpy as np


def get_corrected_img(cv_img1, cv_img2):
    MIN_MATCHES = 10
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(cv_img1, None)
    kp2, des2 = orb.detectAndCompute(cv_img2, None)
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # 根据 Lowe 的比率测试来过滤良好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        print(src_points.shape)
        good_matches = sorted(good_matches, key=lambda x: x.distance)  # 前50匹配点
        match_img = cv2.drawMatches(cv_img1, kp1, cv_img2, kp2, good_matches[:50], None)
        cv2.imshow('flannMatches', match_img)
        cv2.imwrite("flannMatch.jpg", match_img)
        cv2.waitKey()
        m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        corrected_img = cv2.warpPerspective(cv_img1, m, (cv_img2.shape[1], cv_img2.shape[0]))
        return corrected_img
    return None


if __name__ == "__main__":
    cv_im1 = cv2.imread('G:\\fan\\2022\object_detection\data\images\image1\image16.jpg')
    cv_im2 = cv2.imread('G:\\fan\\2022\object_detection\data\images\image1\image17.jpg')
    img = get_corrected_img(cv_im2, cv_im1)
    if img is not None:
        cv2.imshow('Corrected image', img)
        cv2.imwrite("corrected_image.jpg", img)

        cv2.waitKey()
