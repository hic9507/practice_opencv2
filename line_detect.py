import cv2, sys
import copy
import numpy as np
from line_detector import detector
import os
import imutils


# def point_gen(M, N, space, num_of_pixel, shape):
def point_gen(M, N, space, num_of_pixel, shape):
    x1, y1 = M[0][0], M[0][1]
    x2, y2 = N[0][0], N[0][1]

    for i in range(1, int(num_of_pixel) + 1):
        if ((x1 != x2) and (y1 == y2)):
            M.insert(0, [x1, y1 - space * i])
            M.append([x1, y1 + space * i])
            N.insert(0, [x2, y2 - space * i])
            N.append([x2, y2 + space * i+i])

        if ((x1 == x2) and (y1 != y2)):
            M.insert(0, [x1 - space * i, y1])
            M.append([x1 + space * i, y1])
            N.insert(0, [x2 - space * i , y2])
            N.append([x2 + space * i, y2])

        if M[0][1] < 0:  ####왼쪽 튀어나감
            del M[0]

        if M[int(len(M) - 1)][0] >= shape[1]:  ####오른쪽 튀어나감
            del M[int(len(M) - 1)]

        if M[0][0] < 0:  ####위쪽 튀어나감
            del M[0]

        if M[int(len(M) - 1)][1] >= shape[0]:  ####아랫쪽 튀어나감
            del M[int(len(M) - 1)]

        if N[0][1] < 0:  ####왼쪽 튀어나감
            del N[0]

        if N[int(len(N) - 1)][0] >= shape[1]:  ####오른쪽 튀어나감
            del N[int(len(N) - 1)]

        if N[0][0] < 0:  ####위쪽 튀어나감
            del N[0]

        if N[int(len(N) - 1)][1] >= shape[0]:  ####아랫쪽 튀어나감
            del N[int(len(N) - 1)]
    # print(M)
    # print(N)
    return M, N



image = cv2.imread('nonbbox.jpg')
image_gray = cv2.imread('nonbbox.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('nonbbox.jpg')
# image_gray = cv2.imread('nonbbox.jpg', cv2.IMREAD_GRAYSCALE)

# for i in os.listdir('./test/'):
#     path = './test/'+i
#     image = cv2.imread(path, cv2.IMREAD_COLOR)
#     image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('s', image)
    #cv2.waitKey()

#image = cv2.imread('line.jpg', image)
#image_gray = cv2.imread('line.jpg', cv2.IMREAD_GRAYSCALE)
#
# blur = cv2.GaussianBlur(image_gray, (3,3), sigmaX=4.5)
#
# k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#
# erosion = cv2.erode(blur, k)
# erosion2 = cv2.erode(erosion, k)
# erosion3 = cv2.erode(erosion2, k)
# edged = cv2.Canny(erosion3, 200, 200)
#
# # contours를 찾아 크기순으로 정렬
# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#
# findCnt = None
#
# # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
# for c in cnts:
#     # print(c)
#     peri = cv2.arcLength(c, True)
#     # print(peri)
#     approx = cv2.approxPolyDP(c, 0.01 * peri, True)
#
#     # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
#     if len(approx) == 4:
#         findCnt = approx
#         break
#
# # 만약 추출한 윤곽이 없을 경우 오류
# if findCnt is None:
#     raise Exception(("Could not find outline."))
#
# output = image.copy()
# # print(output.shape)
# cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)
# # print(image.shape[0])
# tmp = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
# tmp.fill(0)
# cv2.drawContours(tmp, [findCnt], -1, (255, 255, 255), 2)

# cv2.imshow('s', tmp)
# cv2.waitKey()

# image = cv2.Canny(image_gray, 80, 150)
# path_dir = 'C:/Users/user/anaconda3/envs/opencv/opencv/test'
# file_list = os.listdir(path_dir)
# image = cv2.imread(path_dir)
# cv2.imshow('file', image)
# cv2.waitKey()
# for i in os.listdir('./test/'):
#     path = './test/'+i
#
#     image = cv2.imread(path, cv2.IMREAD_COLOR)
#     image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     # image_gray = cv2.resize(image_gray, (1280, 720))
#     cv2.imshow( "image", image_gray)
#     cv2.waitKey(0)


#### 검은 배경에 사각형
# image_gray = np.zeros(image_gray.shape, np.uint8)
# image = np.zeros(image.shape)
#
# space_test = 20 * 3
# cv2.line(image_gray, (81, 63 + space_test), (1035, 63), (255, 255, 255), 1)
# cv2.line(image_gray, (81, 63 + space_test), (81, 679), (255, 255, 255), 1)
# cv2.line(image_gray, (81, 679), (1035, 679), (255, 255, 255), 1)
# cv2.line(image_gray, (1035, 63), (1035, 679), (255, 255, 255), 1)
#
# cv2.line(image, (81, 63 + space_test), (1035, 63), (255, 255, 255), 1)
# cv2.line(image, (81, 63 + space_test), (81, 679), (255, 255, 255), 1)
# cv2.line(image, (81, 679), (1035, 679), (255, 255, 255), 1)
# cv2.line(image, (1035, 63), (1035, 679), (255, 255, 255), 1)

# 직선의 방정식 equation of a straight line
# x1, y1, x2, y2 = 81, 63, 1035, 63
# x3, y3, x4, y4 = 81, 679, 1035, 679

# x1, y1, x2, y2 = 19, 103, 144, 94
# x3, y3, x4, y4 = 39, 280, 151, 272

# print(image.shape[0])
# print(image.shape[1])

#
x1, y1 = 0, 0
x2, y2 = image.shape[1]-1, 0
x3, y3 = 0, image.shape[0]-1
x4, y4 = image.shape[1]-1, image.shape[0]-1


# 이럴까봐 이걸 변수화 해둔거임
space = 5
w_space = 3/5
h_space = 3/5
num_of_pixel = 30  ##정확히는, 기준 픽셀 위로, 아래로 각각 num_of_pixel개 생성됨
# w_num_of_pixel = 3/5*image_gray.shape[1]
# h_num_of_pixel = 3/5*image_gray.shape[0]
# print(image.shape)

A = [[x1, y1]]
B = [[x2, y2]]
C = [[x3, y3]]
D = [[x4, y4]]

# A, C = point_gen(A, C)

Z, X = point_gen(copy.deepcopy(A), copy.deepcopy(B), space, num_of_pixel,image.shape)
print(detector(Z, X, image, image_gray))

Z, X = point_gen(copy.deepcopy(A), copy.deepcopy(C), space, num_of_pixel,image.shape)
print(detector(Z, X, image, image_gray))

Z, X = point_gen(copy.deepcopy(B), copy.deepcopy(D), space, num_of_pixel, image.shape)
print(detector(Z, X, image, image_gray))

Z, X = point_gen(copy.deepcopy(C), copy.deepcopy(D), space, num_of_pixel, image.shape)
print(detector(Z, X, image, image_gray))

cv2.imshow('dasd', image)
k = cv2.waitKey()
while(1):
    if k == 27:
        break


