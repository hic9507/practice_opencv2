import math

import cv2, sys
from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint
import sys, os
# import linear_equation
from collections import namedtuple




image = cv2.imread('C:/Users/user/Desktop/nonbbox.jpg')
image_gray = cv2.imread('C:/Users/user/Desktop/nonbbox.jpg', cv2.IMREAD_GRAYSCALE)

# image = cv2.resize(image, (80, 80))
# image_gray = cv2.resize(image_gray, (80,80))

blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)
ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

edged = cv2.Canny(blur, 40, 80) #이거는 더 조정해도 별 의미 없ㅂㅂ을것ㅂ 같고
# 이거 코드 연구실 학생이 짠거야? 이거? 엣지? ㅇㅇ 이거 내가한겨
# ㅋㅋㅋ 아그려?
# 더 두껍게 나오게 할 수 있어?  ㄱ커ㄴㄹ사이즈 조정하면 되나 킁게ㅚ곽선? 커널 사이즈 조정하면 가우시안 블러랑 다 들어가서 될겨 아 근데 두껍게?
# 는 모르겠네
# edged = cv2.Canny(blur, 10, 250) ㅈㄴ 무섭게 생겼네 근데 암것도 안만졌는데 왜 다르지?

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)) # cv2.getStructuringElement(shape, ksize, [,anchor])

closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closed', closed) ################################################### 2가 선이 선명함
for i in range(len(np.array(closed))):
    for j in range(len(np.array(closed[0]))):
        print(closed[i][j],end=" ")
cv2.waitKey()

# cv2.imshow('closed', closed) ################################################### 2가 선이 선명함
# # print('*'*100)x
# for i in range(len(np.array(closed))):
#     for j in range(len(np.array(closed[0]))):
#         print(closed[i][j],end=" ")
#         print("")
#     print()
# # pprint(np.array(closed2))
# cv2.waitKey()
# # print('*'*100)

# y = ax + b

# (x,y)
# ((x-w), ()) (x+2)
#
# func= (x1-x2)/(y1-y2)X+b

from fractions import Fraction
import sympy

# 직선의 방정식 equation of a straight line
x1, y1, x2, y2 = 81, 63, 1035, 679
a = (y1 - y2) / (x1 - x2)
b = y1 - (a * x1)
X = sympy.symbols("X")

func = a*X + b
previous_count = 0
count = 0

# 아 ㅅㅂ
# 점과 직선 사이의 거리 distance between a point and a straight line
def dist(P, A, B):
    area = abs((A[0] - P[0]) * (B[1] - P[1]) - (A[1] - P[1]) * (B[0] - P[0]))
    AB = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
    return (area / AB)
# print(area.ndim)  # 15552
# print(AB.ndim)  # 1135.5932370351632



#[(81, 61), (81, 62), (81, 63), (81, 64), (81, 65)] ### LT
#[(1035, 61), (1035, 62), (1035, 63), (1035, 64), (1035, 65)] ### RT
#[(79, 677), (80, 679), (81, 679), (81, 679), (82, 679)] ### LB
#[(1035, 677), (1035, 678), (1035, 679), (1035, 680), (1035, 681)] ### RB


# 이럴까봐 이걸 변수화 해둔거임
space = 20
A = [(81, 63-space*3), (81, 63-space*2), (81, 63-space), (81, 63), (81, 63+space), (81, 63+space*2), (81, 63+space*3)]
B = [(1035, 63-space*3), (1035, 63-space*2), (1035, 63-space), (1035, 63), (1035, 63+space), (1035, 63+space*2), (1035, 63+space*3)]
# A = [(81, 61), (81, 62), (81, 63), (81, 64), (81, 65)] ### LT
# B = [(1035, 61), (1035, 62), (1035, 63), (1035, 64), (1035, 65)] ### RT


for i in range(len(A)):
    image = cv2.circle(image, ( int(A[i][0]),  int(A[i][1])), 4, (0,255,0),1) # 이게 픽셀찍은거지 ㅇㅇㅇ 근데 왜 7개가 나와?5갠줄 ㅄ이라 이햊모 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
    image = cv2.circle(image, ( int(B[i][0]),  int(B[i][1])), 4 , (0,255,0),1)


cv2.imshow('aaa', image)
cv2.waitKey()

num_of_line = len(A)*len(B)
cnt_under_one = [0 for i in range(num_of_line)]


########## 이 두 for문은 25가지 대각선 고르기 위한 for문
for q in range(len(A)):
    for w in range(len(B)):

# 이게 지금 0번쨰 인덱스가 젤 좋게 나오는데 그러면 그 뒤로 값이 다 똑같다는거 아니야?
# 카운트값ㅂ? 베스트 라인 인덱스 값 아 카운트를 거기에 저장했나? 엥 그럼 카운트 값이 ㄷ똑같단 소리네
        print("num : ",q * len(A) + w) #이게 지금 46 47 48이지

        cnt_under_one[q *len(A) + w]=0
        ######### 이거는 한 대각선에 대해서 255점과 거리 계산하기 위한 for문
        for i in range(closed.shape[0]):
            for j in range(closed.shape[1]):

                if closed[i][j] == 255:

                    d = dist((i,j), A[q], B[w])
                    # print(d)
                    # print(d)
                    if d < 3:
                        # print(d)
                        cnt_under_one[q*len(A) + w] += 1 #각 대각선(25개)에 대한 흰점<1 의 카운트값 저장

        print("cnt_under_one : ",cnt_under_one[q*len(A) + w]) # 68 이고 이게 ㅇㅇㅇ


    # for i in range(q*5 + w + 1):
#     print("cnt_under_zero[",i, "] : ",cnt_under_zero[i]) 아 여기있네  ㅌ익텉ㄹ 주석처리한거구나


cnt_max = max(cnt_under_one) #이거는 아까 리스트 최댓값
index = cnt_under_one.index(cnt_max) # 이거는 위 리스트 인덱스

print("best line index : ",index)#아니근데 왜 자꾸 40이 나와 ㅅㅂ럼  40이 나오는껀 마ㅉ아 ㅋㅋㅋ 젤크네 ㅄ~

 #이거는 for 문이랑 같은 원리라서 나중에 하는겅고
qq = int(math.trunc(float(index/len(A)))) # qq는 A 좌표 중 몇 번째 인지(0~4 중) 얘는 왜 나누고
ww = int(index%len(A)) # B좌표 중 몇 번째 인지 (0~4 중) 얘는 왜 나머지야?
# qq = (len(A)-1)-int(math.trunc(float(index/len(A)))) # qq는 A 좌표 중 몇 번째 인지(0~4 중) 얘는 왜 나누고
# ww = (len(B)-1)-int(index%len(A)) # B좌표 중 몇 번째 인지 (0~4 중) 얘는 왜 나머지야?

print("test") #아 프린트는 이해해버렸노
print(int(math.trunc(float(index/len(A))))) # 내림 이라 버리고 ? 근데 이거 마지막에 정수로 감싸준거아니야? 5구나 ㅇㅋ
print(math.trunc(float(index/len(A)))) # 이건 trunc로 버리ㅕ서 4
print(float(index/len(A))) # ㄴㄴ 5 이건 플로트 ㅇㅋ
print(index/len(A)) #그냥ㅇ ㅜ언래 값 ㅇㅋㅇㅋㅇㅋㅇㅋㅇㅋㅇㅋㅋ

# 아 지금 저게 q,w가 아니고 qq ww지?ㅇㅇㅇ 오 이건 포문 그 원리노 ㅁㄹ겠다 ㅇㄱ야~ 5-1 ㅋㅋ ㅋ정수 소수점 버리고 인덱스에서 5나눈걸 뺸거네?
# ㅇㅇㅇ 사실 꼳 빼야하는건 아닌데
# 이게 쉽게 말하면
# 이미지는 좌표가 반대니까>??? ㅇㅇ ㅇ그렇지 이해석사
# 지렸고 이건 봐야할듯 아 ㅋㅋㅋㅋ ㅇㅋ
print(qq, ww)
cv2.line(image, (A[qq][0], A[qq][1]), (B[ww][0], B[ww][1]), (255,0,0), 3) # 이거는 [qq]의 0번쨰에서 1번쨰로 긋고 맞지?



# 된다 색 바꾼거 굳

##이건 원래 라이 (form YOLO) = red
cv2.line(image, (A[3][0], A[3][1]), (B[3][0], B[3][1]), (0,0,255),3) # 이거는 [qq]의 0번쨰에서 1번쨰로 긋고 맞지?
#아까 cnt_뭐암튼 : 이거 지웠어?


cv2.imshow('image', image)
cv2.waitKey()


##이 아이는 일단 무시 나중에

if count > previous_count:
    previous_count = count
    print("-----------------count print----------------------")
    print(count)
    print("-----------------previous_count print----------------------")
    print(previous_count)

# print(closed[i][j])
# def printsave(*w):
#     file = open('C:/Users/user/Desktop/stdout.txt', 'w')
#     print(*w)
#     print(*w, file=file)
#     file.close()
# sys.stdout = open('C:/Users/user/Desktop/output.txt', 'w')
# print('-------------------------------- closed[i][j] 출력 --------------------------------')
# print(closed[i][j])
# print()
# print('-------------------------------- i, j 출력 --------------------------------' )
# print(i, j)
# print()
# print('-------------------------------- dist(P, A[i], B[i] 출력 --------------------------------')
# for i in range(0, len(A)):
#     print(dist(P, A[i], B[i]))
# dict.get(closed[i])
                                                                                                # print(dict.get(closed[i]))
                                                                                                # print(closed[i][j])
                                                                                                # print(i,j)
                                                                                                # if
                                                                                                # dict.get('closed[i]' distance)
