import cv2
import imutils
import numpy as np
import os, sys

# for i in os.listdir('C:/Users/user/Desktop/env/LDC/result/CLASSIC2CLASSIC/fused/'):
#     path = 'C:/Users/user/Desktop/env/LDC/result/CLASSIC2CLASSIC/fused/' + i
for i in os.listdir('C:/Users/user/Desktop/frame_remove_back/'):
    path = 'C:/Users/user/Desktop/frame_remove_back/' + i


    image = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (700,800))
    gray = cv2.resize(gray, (700,800))
    if image is None:
        print('Image load failed')
        sys.exit()

# image = cv2.imread('line.jpg')
#     gray = cv2.imread('line.jpg', cv2.IMREAD_GRAYSCALE)

    # blur = cv2.GaussianBlur(image, (15,15), sigmaX=0.1)

    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    #
    # gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, k)
    #
    # merged = cv2.resize(gray, (700, 800))
    #
    # erosion = cv2.dilate(merged, k)
    # erosion2 = cv2.dilate(erosion, k)
    # erosion3 = cv2.erode(erosion2, k)
    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # erosion4 = cv2.erode(erosion3, k)
    # edged = cv2.Canny(gray, 50, 150)

    # erosion = cv2.erode(blur, k)
    # erosion2 = cv2.dilate(erosion, k)
    # erosion3 = cv2.erode(erosion2, k)
    # edged = cv2.Canny(erosion, 150, 150)
    # cv2.imshow('1', edged)
    # cv2.waitKey(0)
    # cv2.imshow('first', image)
    # cv2.imshow('aa', blur)
    # cv2.imshow('bb', edged)
    # cv2.waitKey()

    # contours를 찾아 크기순으로 정렬
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (1000,800))
    gray = cv2.resize(gray, (1000,800))

    blur = cv2.GaussianBlur(gray, (11, 11), sigmaX=0.2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    erosion = cv2.dilate(blur, k)
    erosion2 = cv2.dilate(erosion, k)
    erosion3 = cv2.dilate(erosion2, k)

    edged = cv2.Canny(erosion2, 170, 170)

    # contours를 찾아 크기순으로 정렬
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = None
    try:

        # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
        for c in cnts:
            # print(c)
            peri = cv2.arcLength(c, True)
            # print(peri)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)

            # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
            if len(approx) == 4:
                findCnt = approx
                break

    # 만약 추출한 윤곽이 없을 경우 오류
        if findCnt is None:
            raise Exception(("Could not find outline."))
    except:
        continue


    output = image.copy()
    # print(output.shape)
    cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)
    # print(image.shape[0])
    tmp = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.uint8)
    tmp.fill(0)
    cv2.drawContours(tmp, [findCnt], -1, (255, 255, 255), 2)

    cv2.imshow('s', tmp)
    cv2.waitKey()
    # cv2.imwrite("C:\\Users\\user\\anaconda3\\envs\\opencv\\opencv\\imwrite2\\" + i + '.jpg', tmp)