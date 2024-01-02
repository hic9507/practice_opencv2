import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import sympy

##### 이미지 변환
image = cv2.imread('C:/Users/user/Desktop/nonbbox.jpg')
image_gray = cv2.imread('C:/Users/user/Desktop/nonbbox.jpg', cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(image_gray, ksize=(7,7), sigmaX=0)

ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

edge = cv2.Canny(blur, 3, 40)  # cv2.Canny(gray_img, threshold1, threshold2) 1을 줄이면 엣지가 많이 검출됨 min, max 순임. 2는 높을수록 외곽만 검출

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))

closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closed', closed)
cv2.waitKey()

#### 직선의 방정식 정의
x1, y1, x2, y2 = 81, 63, 1035, 679
a = (y1 - y2) / (x1 - x2)
b = y1 - (a * x1)
X = sympy.symbols("X")
func = a*X + b

#### 점과 직선 사이의 거리 정의
def dist(P, A, B):
    area = abs((A[0] - P[0]) * (B[1] - P[1]) - (A[1] - P[1]) * (B[0] - P[0]))
    AB = ((A[0] - B[0]) ** 2 + (A[1] - B[1] ** 2) ** 0.5)
    return (area / AB)

#### A, B 좌표 정의
space = 20
# A = [(81, 63-space*3), (81, 63-space*2), (81, 63-space), (81, 63), (81, 63+space), (81, 63+space*2), (81, 63+space*3)]
# B = [(1035, 63-space*3), (1035, 63-space*2), (1035, 63-space), (1035, 63), (1035, 63+space), (1035, 63+space*2), (1035, 63+space*3)]
A = [(81, 61), (81, 62), (81, 63), (81, 64), (81, 65)] ### LT
B = [(1035, 61), (1035, 62), (1035, 63), (1035, 64), (1035, 65)] ### RT

for i in range(len(A)):
    image = cv2.circle(image, ( int(A[i][0]),  int(A[i][1])), 4, (0,255,0),1)
    image = cv2.circle(image, ( int(B[i][0]),  int(B[i][1])), 4 , (0,255,0),1)
cv2.imshow('pixel', image)
cv2.waitKey()

#### 5개 좌표에 대한 픽셀에 대해 직선을 긋는 경우의 수, 0보다 작은 카운트 수를 num_of_line 수만큼 실행하여 저장함.
num_of_line = len(A) * len(B)
cnt_under_one = [0 for i in range(num_of_line)]

#### 25가지 대각선을 고르기 위한 for문 정의
for q in range(len(A)):
    for w in range(len(B)):
        print("num : ", q * len(A) + w)

        cnt_under_one[q * len(A) + w] = 0

        #### 한 대각선에 대해서 255인 픽셀의 점과 거리를 계산하기 위한 for문
        for i in range(closed.shape[0]):
            for j in range(closed.shape[1]):
                if closed[i][j] == 255:
                    d = dist((i,j), A[q], B[w])
                    # print(d) (0.2654551713707589-1.9115075820009792e-05j) 이런게 엄청 나옴
                    if d < 1:
                        cnt_under_one[q * len(A) + w] # 각 대각선(25개)에 대한 흰 점의 길이가 < 1 인 것 카운트하여 저장
        print("cnt_under_one : ", cnt_under_one[q*len(A) + w])

cnt_max = max(cnt_under_one)
index = cnt_under_one.index(cnt_max)

print("best line index : ", index)

qq = int(math.trunc(float(index/len(A))))
ww = int(index%len(A))

print('qq, ww print')
print(qq, ww)

cv2.line(image, (A[qq][0], A[qq][1]), (B[ww][0], B[ww][1]), (255,0,0), 3)

#### 원래 YOLO 바운딩 박스 라인 = 빨간색
cv2.line(image, (A[3][0], A[3][1]), (B[3][0], B[3][1]), (0,0,255),3)
cv2.imshow('image', image)
cv2.waitKey()

# if count > previous_count:
#     previous_count = count
#     print("-----------------count print----------------------")
#     print(count)
#     print("-----------------previous_count print----------------------")
#     print(previous_count)