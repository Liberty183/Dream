import os
import cv2 as cv
from imutils.perspective import four_point_transform
import imutils
from imutils import contours

PATH = 'raw'

opo = 0
for k in os.listdir(PATH):
    q = os.path.join(PATH, k)

    gray = cv.cvtColor(cv.imread(q), cv.COLOR_BGR2GRAY)

    gray = cv.threshold(gray, 135, 255, cv.THRESH_BINARY)[1]
    cont = sorted(cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0], key=cv.contourArea)[-2]

    cont = cv.approxPolyDP(cont, 0.02 * cv.arcLength(cont, True), True).reshape(4, 2)

    warp = four_point_transform(gray, cont)
    w = warp.shape[1]
    h = warp.shape[0]
    for row in range(33):
        for col in range(10):
            opo += 1
            num = warp[row * h // 33: (row + 1) * h // 33, col * w // 10: (col + 1) * w // 10]
            num = cv.resize(num, (50, 50), cv.INTER_AREA)
            cv.imwrite(f'dataset\\{col}\\{opo}.png', num)
        print(k, row)
    warp = cv.resize(warp, (warp.shape[1] // 3, warp.shape[0] // 3))
nkjjl