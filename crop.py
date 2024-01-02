import cv2
import numpy as np
import imutils

image = cv2.imread('C:/Users/user/Desktop/p.jpg')
image = cv2.resize(image, (350, 350))
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# cv2.imshow('image_gray', image_gray_blurred)
# cv2.waitKey(0)

ret,thresh1 = cv2.threshold(image_gray_blurred,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(image_gray_blurred,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(image_gray_blurred,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(image_gray_blurred,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(image_gray_blurred,127,255,cv2.THRESH_TOZERO_INV)

images_row1 = np.hstack([image_gray_blurred, thresh1, thresh2])
images_row2 = np.hstack([thresh3, thresh4, thresh5])
images_combined = np.vstack((images_row1, images_row2))

# cv2.imshow('Images', images_combined )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape

sd = ShapeDetector()
cnts = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
    if sd.detect(c) != 'rectangle': next
    c = c.astype("float")
    c = c.astype("int")
    x,y,w,h = cv2.boundingRect(c)

    cv2.rectangle(thresh2,(x,y),(x+w,y+h),(3,255,4),2)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
