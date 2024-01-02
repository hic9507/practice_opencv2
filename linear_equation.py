from fractions import Fraction
import sympy
import numpy as np
import cv2

# 직선의 방정식
x1, y1, x2, y2 = 81, 63, 1035, 679
a = (y1 - y2) / (x1 - x2)
b = y1 - (a * x1)
X = sympy.symbols("X")

func = a*X + b
previous_count = 0
count = 0

# 점과 직선 사이의 거리
def dist(P, A, B):
    area = abs((A[[0]] - P[[0]]) * (B[[1]] - P[[1]]) - (A[[1]] - P[[1]]) * (B[[0]] - P[[0]]))
    AB = ((A[[0]] - B[[0]]) ** 2 + (A[[1]] - B[[1]]) ** 2) ** 0.5
    return (area / AB)

# A = (81, 61)
# B = (1035, 679)
# P = (0, 25)

# print(dist(P, A, B))
# A0, A1, A2, A3, A4 = [(81, 61), (81, 62), (81, 63), (81, 64), (81, 65)]
# B0, B1, B2, B3, B4 = [(1035, 677), (1035, 678), (1035, 679), (1035, 680), (1035, 681)]
# A = np.array([[A0, A1, A2, A3, A4]])
# B = np.array([[B0, B1, B2, B3, B4]])

a = np.array([[[[1,2], [2,4]]]])

# print(A)
# print(B)
# print(type(A))
# print(A.ndim)
# print(a.ndim)

space = 20
A = [(81, 63-space*3), (81, 63-space*2), (81, 63-space), (81, 63), (81, 63+space), (81, 63+space*2), (81, 63+space*3)]
B = [(1035, 63-space*3), (1035, 63-space*2), (1035, 63-space), (1035, 63), (1035, 63+space), (1035, 63+space*2), (1035, 63+space*3)]

img = np.zeros((512, 512,3 ), np.uint8)
img = cv2.resize(img, (1280,720))
# img = cv2.resize(img, (80,80))
cv2.imshow('img', img)
cv2.waitKey()
print(img.shape)
# cv2.rotatedRectangleIntersection()

# for i in range(len(A)):
#     img = cv2.circle(img, (int(A[i][0]),  int(A[i][1])), 4, (255, 255, 255), 1)
#     img = cv2.circle(img, (int(B[i][0]),  int(B[i][1])), 4 , (255, 255, 255), 1)
# cv2.imshow('aaa', img)
# cv2.waitKey()
# # cv2.line(image, (A[3][0], A[3][1]), (B[3][0], B[3][1]), (0,0,255),3)
# h = int(A[i][0])
# w = int(B[i][0])
# cv2.rectangle(img, (A[3][0], A[3][1]), (B[3][0], B[3][1]), (255,255,255), 1)
# cv2.imshow('aaaaa', img)
# cv2.waitKey()

h = img.shape[0]
w = img.shape[1]
cv2.rectangle(img, (81, 63), (1035, 679), (255,255,255), 1)
cv2.imshow('aaaaa', img)
# cv2.imwrite('C:/Users/user/Desktop/rectancle1.jpg', img)
cv2.waitKey()
print(img.shape)

# for i in range(len(np.array(img))):
#     for j in range(len(np.array(img[0]))):
#         print(img[i][j],end=" ")
# cv2.waitKey()
# cv2.imshow('C:/Users/user/Desktop/rectancle1.jpg')
print()