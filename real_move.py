import cv2
import numpy as np
import copy
import sys
from real_line_detector import detector

##### 픽셀 표시해주는 함수 정의 : (M,N)=(x,y), space=픽셀 간 간격, num_of_pixel=생성되는 픽셀의 개수
def pixel_check(M, N, space, num_of_pixel, shape):
    x1, y1 = M[0][0], M[0][1]
    x2, y2 = N[0][0], N[0][1]

    for i in range(1, num_of_pixel+1): # 1부터 시작해서 +1

        if( (x1 != x2) and (y1==y2) ): # x1!=x2, y1=y2 >> AB or CD
            M.insert(0, [x1, y1 - space * i]) # 0번째 인덱스에 이 값을 계속 반복해서 넣어줌
            M.append([x1, y1 + space * i])    # 리스트에 차례대로 값들을 반복해서 넣어줌
            N.insert(0, [x2, y2 - space * i]) # 0번째 인덱스에 이 값을 계속 반복해서 넣어줌
            N.append([x2, y2 + space * i])

        if( (x1 == x2) and (y1 != y2) ): # x1=x2, y1!=y2 >> AC or BD
            M.insert(0, [x1 - space * i, y1]) # 0번째 인덱스에 이 값을 계속 반복해서 넣어줌
            M.append([x1 + space * i, y1])    # 리스트에 차례대로 값들을 반복해서 넣어줌
            N.insert(0, [x2 - space * i, y2]) # 0번째 인덱스에 이 값을 계속 반복해서 넣어줌
            N.append([x2 + space * i, y2])    # 리스트에 차례대로 값들을 반복해서 넣어줌


##### M값에 대해서 사진 밖으로 튀어나가는 픽셀 삭제. 사진보다 크면 화면에 픽셀이 안나타나는데 거기서 찾으라하니까 에러가 남 실제 존재하지 않는 값을 호출을 하니까 에러가 나서 삭제함
        if M[0][1] < 0:                       ####왼쪽 튀어나감
            del M[0]

        if M[int(len(M) - 1)][0] > shape[1]:  ####오른쪽 튀어나감 -1 ?? for문 range랑 같은 원리??
            del M[int(len(M) - 1)]

        if M[0][0] < 0:                       ####위쪽 튀어나감
            del M[0]

        if M[  int(len(M) - 1)  ][1] > shape[0]:  ####아랫쪽 튀어나감 int 안쓰면 에러남 렝스함수 계산 자료형이 뭔지 모름
            del M[int(len(M) -1)]

##### N값에 대해서 사진 밖으로 튀어나가는 픽셀 삭제
        if N[0][1] < 0:                       ####왼쪽 튀어나감
            del N[0]

        if N[int(len(N) - 1)][0] > shape[1]:  ####오른쪽 튀어나감
            del N[int(len(N) -1)]

        if N[0][0] < 0:                       ####위쪽 튀어나감
            del N[0]

        if N[int(len(N) - 1)][1] > shape[0]:  ####아랫쪽 튀어나감
            del N[int(len(N) -1)]


    return M, N

##### 이미지 불러오기
image = cv2.imread('aa.jpg')
image_gray = cv2.imread('aa.jpg', cv2.IMREAD_GRAYSCALE)
#
# ##### 검은 배경에 사각형 그리기
# image_gray = np.zeros(image_gray.shape, np.uint8)
# image = np.zeros(image.shape)
#
# space_test = 20*3
# cv2.line(image_gray, (81, 63+space_test), (1035, 63), (255, 255, 255), 5)
# cv2.line(image_gray, (81, 63+space_test), (81, 679), (255, 255, 255), 5)
# cv2.line(image_gray, (81, 679), (1035, 679), (255, 255, 255), 5)
# cv2.line(image_gray, (1035, 63), (1035, 679), (255, 255, 255), 5)
#
# cv2.line(image, (81, 63+space_test), (1035, 63), (255, 255, 255), 5)
# cv2.line(image, (81, 63+space_test), (81, 679), (255, 255, 255), 5)
# cv2.line(image, (81, 679), (1035, 679), (255, 255, 255), 5)
# cv2.line(image, (1035, 63), (1035, 679), (255, 255, 255), 5)

##### (x1,y1), (x2,y2), (x3,y3), (x4,y4) 정의
x1, y1, x2, y2 = 81, 63, 1035, 63       # 좌상단, 우상단
x3, y3, x4, y4 = 81, 679, 1035, 679     # 좌하단, 우하단

##### 픽셀 간 간격, 생성될 픽셀의 개수에 대한 변수 정의
space = 10        # 픽셀 간 간격
num_of_pixel = 5  # 기준 픽셀 위,아래로 각각 생성될 픽셀의 개수

A = [[x1, y1]]
B = [[x2, y2]]
C = [[x3, x4]]
D = [[x4, y4]]

Z, X = pixel_check(copy.deepcopy(A), copy.deepcopy(B), space, num_of_pixel, image.shape) #shape이 왜 쓰였냐. 위에서 이미지 shape만큼의 영역에서만 데이터를 만들고 추가하는데, 이게 없으면 의미가 없으니 오류 발생
print(detector(Z, X, image, image_gray))

Z, X = pixel_check(copy.deepcopy(A), copy.deepcopy(C), space, num_of_pixel, image.shape)
print(detector(Z, X , image, image_gray))

Z, X = pixel_check(copy.deepcopy(B), copy.deepcopy(D), space, num_of_pixel, image.shape)
print(detector(Z, X, image, image_gray))

Z, X = pixel_check(copy.deepcopy(C), copy.deepcopy(D), space, num_of_pixel, image.shape)
print(detector(Z, X, image, image_gray))
