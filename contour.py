# __author__ = 'samsjang@naver.com'

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('nonbbox.jpg', cv2.IMREAD_COLOR)
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    cv2.drawContours(image, [contours[i]], 0, (0, 0, 255), 2)
    cv2.putText(image, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
    print(i, hierarchy[0][i])
    cv2.imshow("src", image)
    cv2.waitKey(0)

##################### 욜로처럼 그냥 바운딩 박스 그리고 객체에 맞춰 바운딩 박스 각도 조절 #####################################
# def bbox():
#     img = cv2.imread('1.jpg')
#     img1 = img.copy()
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     ret, thr = cv2.threshold(img_gray, 127, 255, 0)  # cv2.threshold(img, threshold_value, value, flag)
#     contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     cnt = contours[0]
#
#     x, y, w, h = cv2.boundingRect(cnt)
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
#
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#
#     cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
#
#     cv2.imshow('rectangle', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # return convex()
# bbox()

############################################# 객체 윤곽선 검출 ###########################################################
def contour():
    img = cv2.imread('1.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(img_gray, 127, 255, 0)  # cv2.threshold(img, threshold_value, value, flag)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, 0, (0, 0, 255), 1)
    cv2.imshow('thresh', thr)
    cv2.imshow('contour', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

contour()


####################################        꼭짓점 검출        #########################################################
# img = cv2.imread('1.jpg')#
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img1 = img.copy()
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_point = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# img_gray = np.float32(img_gray)
# result = cv2.cornerHarris(img_gray, 2, 3, 0.04)
# result = cv2.dilate(result, None, iterations=6)
# img_point[result>0.01*result.max()]=[255,0,0]
#
# plt.subplot(1,2,1)
# plt.imshow(img_rgb)
# plt.xticks([])
# plt.yticks([])
#
# plt.subplot(1, 2, 2)
# plt.imshow(img_point)
# plt.xticks([])
# plt.yticks([])
#
# plt.show()
####################################        꼭짓점 검출        #########################################################