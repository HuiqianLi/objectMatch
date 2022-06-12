import cv2

cv_img1 = cv2.imread('G:\\fan\\2022\object_detection\data\images\image1\image16.jpg', 1)
cv_img2 = cv2.imread('G:\\fan\\2022\object_detection\data\images\image1\image17.jpg', 1)
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(cv_img1, None)
kp2, des2 = orb.detectAndCompute(cv_img2, None)
# matcher接受normType，它被设置为cv2.NORM_L2用于SIFT和SURF, cv2.NORM_HAMMING用于ORB, FAST and BRIEF
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)  # 前50匹配点
match_img = cv2.drawMatches(cv_img1, kp1, cv_img2, kp2, matches[:50], None)
cv2.imshow('Matches', match_img)
cv2.imwrite("Match.jpg", match_img)
cv2.waitKey()
