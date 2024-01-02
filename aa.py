import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys, os
from fractions import Fraction
import sympy

##### 이미지 변환
image = cv2.imread('C:/Users/user/Desktop/rectancle1.jpg')
image_gray = cv2.imread('C:/Users/user/Desktop/rectancle1.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (1280,720))

height, width = image_gray.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), -5, 1)
dst = cv2.warpAffine(image_gray, matrix, (width, height))

blur = cv2.GaussianBlur(dst, ksize=(5,5), sigmaX=0)

ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

edged = cv2.Canny(blur, 3, 50)
# edged = cv2.Canny(blur, 10, 250)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)) # cv2.getStructuringElement(shape, ksize, [,anchor])

closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

# cv2.imshow('closed', closed) ################################################### 2가 선이 선명함
cv2.imshow('closed', closed)
for i in range(len(np.array(closed))):
    for j in range(len(np.array(closed[0]))):
        print(closed[i][j],end=" ")
cv2.waitKey()
print('\n')

# 직선의 방정식 equation of a straight line
x1, y1, x2, y2 = 81, 63, 1035, 679
a = (y1 - y2) / (x1 - x2)
b = y1 - (a * x1)
X = sympy.symbols("X")
func = a*X + b
previous_count = 0
count = 0

# 점과 직선 사이의 거리 distance between a point and a straight line
def dist(P, A, B):
    area = abs((A[0] - P[0]) * (B[1] - P[1]) - (A[1] - P[1]) * (B[0] - P[0]))
    AB = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
    return (area / AB)

space = 10
A = [(81, 63-space*3), (81, 63-space*2), (81, 63-space), (81, 63), (81, 63+space), (81, 63+space*2), (81, 63+space*3)]
B = [(1035, 63-space*3), (1035, 63-space*2), (1035, 63-space), (1035, 63), (1035, 63+space), (1035, 63+space*2), (1035, 63+space*3)]
# A = [(231, 117-space*2),(231, 117-space), (231, 117), (231, 117+space), (231, 117+space)*2]
# B = [(989, 117-space*2),(989, 117-space), (989, 117), (989, 117+space), (989, 117+space)*2]
for i in range(len(A)):
    dst = cv2.circle(dst, (int(A[i][0]),  int(A[i][1])), 4, (0, 255, 0), 1)
    dst = cv2.circle(dst, (int(B[i][0]),  int(B[i][1])), 4 , (0, 255, 0), 1)

cv2.imshow('aaa', dst)
cv2.waitKey()

num_of_line = len(A)*len(B)
cnt_under_one = [0 for i in range(num_of_line)]


########## 이 두 for문은 25가지 대각선 고르기 위한 for문
for q in range(len(A)):
    for w in range(len(B)):

        print("num : ",q * len(A) + w)

        cnt_under_one[q *len(A) + w]=0
        ######### 이거는 한 대각선에 대해서 255점과 거리 계산하기 위한 for문
        for i in range(closed.shape[0]):
            for j in range(closed.shape[1]):

                if closed[i][j] == 255:

                    d = dist((i,j), A[q], B[w])
                    if d < 3:
                        # print(d)
                        cnt_under_one[q*len(A) + w] += 1 #각 대각선(25개)에 대한 흰점<1 의 카운트값 저장

        print("cnt_under_one : ",cnt_under_one[q*len(A) + w])

cnt_max = max(cnt_under_one) # 리스트 최대 인덱스 값
index = cnt_under_one.index(cnt_max) # 이거는 위 리스트 최대 인덱스를 index에 저장

print("best line index : ",index)

#### xy좌표가 반대로 되어있어서
qq = len(A)-1 - int(math.trunc(float(index/len(A))))
ww = len(B)-1 - int(index%len(A))
# print("test")
# print(int(math.trunc(float(index/len(A))))) # 5
# print(math.trunc(float(index/len(A)))) # 4
# print(float(index/len(A))) # float
# print(index/len(A)) # float

print('qq, ww 출력')
print(qq, ww)
# print(len(cnt_max))
# for z in range(len(A)*(len(B))):
#     for x in range(len(cnt_max)):
# cv2.line(dst, (A[3][0], A[3][1]), (B[3][0], B[3][1]), (255,0,0), 3)
cv2.line(dst, (A[qq][0], A[qq][1]), (B[ww][0], B[ww][1]), (255,0,0), 3)


##이건 원래 라인 (form YOLO) = red
# cv2.line(image, (A[3][0], A[3][1]), (B[3][0], B[3][1]), (0,0,255),3)

cv2.imshow('image', dst)
cv2.waitKey()