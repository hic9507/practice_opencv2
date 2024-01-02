import cv2
import numpy as np
import os
a = 70
b = 255

for i in os.listdir('C:/Users/user/anaconda3/envs/craft-pytorch/deep-text-recognition-benchmark/demo_image/'):
    path = 'C:/Users/user/anaconda3/envs/craft-pytorch/deep-text-recognition-benchmark/demo_image/' + i

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # cv2.imshow('binary', dst)
    # cv2.waitKey()

    cv2.imwrite('C:/Users/user/Desktop/demo_image/' + i, dst)
    # cv2.imshow('1', img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 1)
    # gray = cv2.imread('nonbbox.jpg', cv2.IMREAD_GRAYSCALE)
    # canny = cv2.Canny(gray, 0, 40 )
    # ret, dst = cv2.threshold(gray, a, b, cv2.THRESH_BINARY)
    # ret, dst = cv2.threshold(gray, a, b, cv2.THRESH_BINARY_INV)
    # dst = cv2.bitwise_not(gray)
    # result = cv2.erode(dst, (9,9), iterations = 5)


    # cv2.imshow('canny', dst)
    # cv2.imshow('result', result)



# kernel = np.ones((5, 5), np.uint8)
# result = cv2.erode(img, kernel, iterations = 15)
# ret, dst = cv2.threshold(gray, a, b, cv2.THRESH_BINARY)
# cv2.imshow('zz', dst)
# cv2.imshow("Result", gray)
# cv2.waitKey(0)
# result = cv2.dilate(img, kernel, iterations = 2)
# result = cv2.dilate(img, kernel, iterations = 5)

# cv2.imshow("Source", img)
