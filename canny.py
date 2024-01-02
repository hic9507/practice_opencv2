import cv2, sys
from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint

image = cv2.imread('C:/Users/user/Desktop/line.jpg')
image_gray = cv2.imread('C:/Users/user/Desktop/line.jpg', cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (80, 80))
image_gray = cv2.resize(image_gray, (80,80))

b, g, r = cv2.split(image)
image2 = cv2.merge([r, g, b])

# plt.imshow(image)
# plt.imshow(image2)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# cv2.imshow('image', image)
# cv2.imshow('image_gray', image_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

blur1 = cv2.GaussianBlur(image_gray, ksize=(3,3), sigmaX=0)
ret, thresh1 = cv2.threshold(blur1, 127, 255, cv2.THRESH_BINARY)
blur2 = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)
ret, thresh2 = cv2.threshold(blur2, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow('blur1', blur1 )
# cv2.imshow('blur2', blur2 )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

edged1 = cv2.Canny(blur1, 10, 250)
edged2 = cv2.Canny(blur2, 10, 250)
# cv2.imshow('Edged1', edged1)
# cv2.imshow('Edged2', edged2)
# cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)) # cv2.getStructuringElement(shape, ksize, [,anchor])
closed1 = cv2.morphologyEx(edged1, cv2.MORPH_CLOSE, kernel) # 모폴로지 연산: 침식, 팽창, 열림, 닫힘
closed2 = cv2.morphologyEx(edged2, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('closed1', closed1)
cv2.imshow('closed2', closed2) ################################################### 2가 선이 선명함
print('*'*100)
for i in range(len(np.array(closed2))):
    for j in range(len(np.array(closed2[0]))):
        print(closed2[i][j],end=" ")
    print()
# pprint(np.array(closed2))
cv2.waitKey()
print('*'*100)
# cv2.waitKey(0)

contours, _ = cv2.findContours(closed2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.findContours(src,mode, method, contours=None,hierarchy=None,offset=None)
# >> src: 입력 이미지, mode: 외곽선 검출 방식(EXTERNAL은 가장 바깥쪽 영역만 추출), method: 외곽선 근사 방법, contours: 외곽선 좌표(np.ndarray)-len(contours)외곽선 갯수
# hierarchy: 외곽선 계층 정보를 담고 있는 (1,n,4) shape의 list(n은 contour 개수), next,previous,,child,parent를 의미, offset: 좌표 이동 offset(default=(0,0))

total = 0

contours_image = cv2.drawContours(image, contours, -1, (255, 0, 255), 2)
# plt.imshow(contours_image)
# plt.show()
# cv2.imshow('contours_image', contours_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours_xy = np.array(contours)
# print(len(contours_xy))         ############### 1
# print(contours_xy.shape)        ############### (1, 357, 1, 2)

# x의 min과 max 찾기
x_min, x_max = 0, 0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0])  #네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)
# print(x_min)           ################ 80
# print(x_max)           ################ 292

# y의 min과 max 찾기
y_min, y_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)
# print(y_min)          ################# 12
# print(y_max)          ################# 375

# image trim 하기
x = x_min
y = y_min
w = x_max-x_min
h = y_max-y_min

img_trim = image[y:y+h, x:x+w]
cv2.imwrite('org_trim.jpg', img_trim)
org_image = cv2.imread('org_trim.jpg')

# cv2.imshow('org_image', org_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

