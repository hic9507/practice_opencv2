import cv2
import copy
import math

##### 점과 직선 사이의 거리 함수 정의
def dist(P, A, B):

    area = abs((A[0] - P[0]) * (B[1] - P[1]) - (A[1] - P[1]) * (B[0] - P[0]))
    AB = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
    # print(area)
    # print(AB)
    return (area / AB)

##### 주어진 바운딩 박스의 점 A, B와 신분증 사진의 변환을 통해 흰 점을 지나는 라인을 탐색하는 함수 정의
def detector(A, B, image, image_gray):
    print("---------- 픽셀 돌아가며 탐색 중 ----------")

    for i in range(len(A)):
        image = cv2.circle(image, (int(A[i][0]), int(A[i][1])), 4, (0, 0, 255), 1)
    for i in range(len(B)):
        image = cv2.circle(image, (int(B[i][0]), int(B[i][1])), 4, (0, 0, 255), 1)

    ##### 이미지 엣지 검출
    blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)

    ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(blur, 14, 50)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    #####
    num_of_line = len(A)*len(B) # 픽셀 간 긋는 대각선의 갯수: 팩토리얼처럼 생각
    cnt_under_one = [0 for i in range(num_of_line)] # ?? 대각선의 개수만큼 0을 초기값으로 저장

    index = 0
    cnt = 0
    pre_cnt = 0

    ##### A, B 좌표에 대한 len(A)*len(B)만큼의 대각선을 고르기 위한 for문 정의
    for q in range(len(A)):
        for w in range(len(B)):

            ##### 픽셀 간 대각선을 그어 흰점을 지나는지 확인할 때 그려지는 탐색 영역 박스 그리기(파란색)
            img = copy.deepcopy(image) ##### img라는 변수에 image를 깊은 복사, 주소를 가져옴
            # img = cv2.rectangle(img, (A[q][0], A[q][1]), (B[w][0], B[w][1]), (255, 0, 0), 2) ##### 위에서 복사한 이미지로 사각형을 그리는데, 시작점과 끝점이 (A[q][0], A[q][1]), (B[w][0], B[w][1])이다. 이것은 정의된 좌표의 첫번째 (x,y)를 뜻함

            # print("num : ", q * len(A) + w) # 대각선이 몇 번 그어졌는지 총 수를 의미함.

            cnt_under_one[q * len(A) + w] = 0 # ?? 위에서 정의된 변수의 (q * len(A) + w)번째 인덱스(A, B가 각 좌표 5개씩 가지면 총 25개) 이게 0이어야 실제 검출한 라인과 동일한 픽셀이 있을 때 0부터 카운팅해야하니 0으로 정의함.
            index = 0
            cnt = 0
            pre_cnt = 0
            ##### 픽셀 간 대각선을 그을 때 한 대각선에 대해 픽셀값이 255인 점(흰점)과의 거리를 계산하기 위한 for문
            if A[q][1] <= B[w][1]: # A의 y표값이 B의 y좌표값보다 작거나 같으면 A는 좌푠데 q는 A의 length니까
                for i in range(A[q][1], B[w][1]+1): # A의 첫 y좌표부터 B의 y좌표까지 범위를 정하고 0부터 시작하므로 +1
                    if A[q][0] <= B[w][0]: # A의 x좌표값이 B의 x좌표값보다 작거나 같으면
                        for j in range(A[q][0], B[w][0]+1): # A의 x좌표값부터 B의 x좌표까지 범위를 정하고 0부터 시작하므로 +1
                            if closed[i][j] == 255: # i=y, j=x, 좌표값이 픽셀값이 255인 곳을 지나면
                                d = dist((j, i), A[q], B[w]) # dist 함수를 d로 다시 정의하고 행렬이니까 x,y위치 제대로 바꾸고 A와 B의 좌표를 순서대로 입력함
                                if d < 3: # 점과 직선 사이의 거리 함수를 재정의한 d, 픽셀 간 그은 대각선과 픽셀값이 255인 지점의 거리가 1 미만이면
                                    img = cv2.circle(img, (j, i), 4, (0, 255, 255), 1) # img 파일에 (j, i)를 원의 중심으로 놓고 반지름 값을 4로 둬서 픽셀 값을 표시함. 물론 픽셀값이 255여야하고 d<1이어야 함.
                                    # cnt_under_one[q * len(A) + w] += 1 # 각 대각선(25개)에 대한 흰점<1의 카운트값 저장
                        if pre_cnt < cnt:
                            pre_cnt = cnt
                            index = q * len(A) + w
                            print(index)

                    else: # A의 x좌표값이 B의 x좌표값보다 작거나 같지 않고 크면
                        for j in range(B[w][0], A[q][0] + 1): # B의 첫 x값부터 A의 첫 x값+1까지 범위를 정하고
                            if closed[i][j] == 255: # i=y, j=x, 위와 동일
                                d = dist((j, i), A[q], B[w]) # 위와 동일
                                if d < 3:# 위와 동일
                                    img = cv2.circle(img, (j, i), 4, (0, 255, 255), 1) # 위와 동일
                                    # cnt_under_one [q * len(A) + w] += 1 # 위와 동일
                        if pre_cnt < cnt:
                            pre_cnt = cnt
                            index = q * len(A) + w
                            print(index)            # result_cnt = cnt_under_one[j]
                                        # cnt_under_one.free()

            else: # A의 y좌표값이 B의 y좌표값보다 크면
                for i in range(B[w][1], A[q][1] + 1): # B의 y좌표값부터 A의 y좌표값까지 범위를 정하고 0부터 시작하므로 +1
                    if A[q][0] <= B[w][0]: # A의 x좌표값이 B의 x좌표값보다 작거나 같으면
                        for j in range(A[q][0], B[w][0] + 1): # A의 x좌표값부터 B의 x좌표값까지 범위를 정하고 0부터 시작하므로 +1
                            if closed[i][j] == 255: # 위와 동일
                                d = dist((j, i), A[q], B[w]) # 위와 동일
                                if d < 3: # 위와 동일
                                    img = cv2.circle(img, (j, i), 4, (0, 255, 255), 1) # 위와 동일
                                    # cnt_under_one[q * len(A) + w] += 1  # 위와 동일
                        if pre_cnt < cnt:
                            pre_cnt = cnt
                            index = q * len(A) + w
                            print(index)


                    else: # A의 x좌표값이 B의 x좌표값보다 크면
                        for j in range(B[w][0], A[q][0] + 1): # B의 x좌표값부터 A의 x좌표값까지 범위를 정하고 0부터 시작하므로 +1
                            if closed[i][j] == 255: # i=y, j=x, 위와 동일
                                d = dist((j, i), A[q], B[w]) # 위와 동일
                                if d < 3:# 위와 동일
                                    img = cv2.circle(img, (j, i), 4, (0, 255, 255), 1) # 위와 동일
                                    # cnt_under_one[q * len(A) + w] += 1 # 위와 동일
                        if pre_cnt < cnt:
                            pre_cnt = cnt
                            index = q * len(A) + w
                            print(index)

            ##### 시각화: 한 대각선 당 검출된 흰 색 픽셀들을 노란색으로 표시
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # print("cnt_under_one : ", cnt_under_one[q * len(A) + w])

    # cnt_max = max(cnt_under_one) # 리스트의 최댓값을 구해서 cnt_max에 저장
    # index = cnt_under_one.index(cnt_max) # 위 cnt_max의 인덱스를 index에 저장

    #####
    qq = int(math.trunc(float(index/len(A)))) # len(A)만큼의 길이의 A 좌표 중 몇 번째인지 qq에 저장
    # qq = index/len(A)
    ww = int(index%len(B)) # len(B)만큼의 길이의 B 좌표 중 몇 번째인지 ww에 저장

    ##### 위의 반복문을 통해 픽셀 간 그은 대각선과 흰 점(픽셀 값=255)이 만나는 부분을 찾은 라인 그리기, >> 찾은 진짜 라인 그리기
    cv2.line(image, (A[qq][0], A[qq][1]), (B[ww][0], B[ww][1]), (255, 0, 0), 3) # image 파일에 찾은 A의 (x,y)좌표부터 B의 (x,y)좌표까지 라인을 긋는다. 처음 y를 [0]으로 놨었음

    ##### YOLOv5를 통해서 얻은 바운딩 박스의 좌표값을 이용하여 바운딩 박스 그리기 = 빨간색
    # A, B가 dist함수에 주어진 욜로의 바운딩박스 좌상단 우하단 좌표이므로 space와 num_of_pixel로 생성된 길이에서 반으로 나누고
    # 각 (x,y) 좌표를 그리는데, y,x 순임
    cv2.line(image, (A[int(len(A)/2)][0], A[int(len(A)/2)][1]), (B[int(len(B)/2)][0], B[int(len(B)/2)][1]), (0, 0, 255), 3) # [0]을 2뒤에 써야 하는데 대괄호 앞에 쓰고 있었음
    # cv2.line(image, (A[int(len(A)/2)][0], A[int(len(A)/2)][1]), (B[int(len(B/2))][0], B[int(len(B)/2)][1]), (0, 0, 255), 3)

    print("탐색 완료, 라인 검출")

    cv2.imshow('image', image)
    while(1):
        k = cv2.waitKey()
        if k == 27: # ESC(아스키코드) 누르면 꺼짐
            break

    return [A[qq][0], A[qq][1]], [B[ww][0], B[ww][1]]