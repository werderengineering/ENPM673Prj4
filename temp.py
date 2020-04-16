import cv2
import numpy as np
import os

import imagesProcessing as ip


def rectAndTemp_problem1(path, flag=False):
    images = []
    cv_img = []
    for image in os.listdir(path):
        images.append(image)
    images.sort()
    for image in images:
        img = cv2.imread("%s/%s" % (path, image))
        cv_img.append(img)

    frame = cv2.imread('./Bolt2/img/0001.jpg')
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # img = cv2.equalizeHist(img)
    xl = 250
    xu = 320
    yl = 75
    yu = 150
    rect = np.array([[xl, xu, xu], [yl, yl, yu], [1, 1, 1]])
    # temp = img[yl:yu, xl:xu]

    temp = ip.subImageInBoundingBoxAndEq(img, rect, histEqualize=True)
    if flag:
        cv2.imwrite('./Bolt2/template.jpg', temp)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
        cv2.imshow("Template", temp)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    return rect, temp, cv_img


def rectAndTemp_problem2(path, flag=False):
    images = []
    cv_img = []
    for image in os.listdir(path):
        images.append(image)
    images.sort()
    for image in images:
        img = cv2.imread("%s/%s" % (path, image))
        cv_img.append(img)

    frame = cv_img[20]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # img = cv2.equalizeHist(img)
    xl = 68
    xu = 160
    yl = 58
    yu = 130
    rect = np.array([[xl, xu, xu], [yl, yl, yu], [1, 1, 1]])
    temp = ip.subImageInBoundingBoxAndEq(img, rect, histEqualize=True)
    if flag:
        cv2.imwrite('./Car4/template.jpg', temp)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
        cv2.imshow("Template", temp)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    return rect, temp, cv_img[20:]


def rectAndTemp_problem3(path, flag=False):
    images = []
    cv_img = []
    for image in os.listdir(path):
        images.append(image)
    images.sort()
    for image in images:
        img = cv2.imread("%s/%s" % (path, image))
        cv_img.append(img)

    frame = cv2.imread('./DragonBaby/img/0001.jpg')
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # img = cv2.equalizeHist(img)
    xl = 130
    xu = 220
    yl = 75
    yu = 300

    rect = np.array([[xl, xu], [yl, yu], [1, 1]])
    temp = ip.subImageInBoundingBoxAndEq(img, rect, histEqualize=True)
    if flag:
        cv2.imwrite('./DragonBaby/template.jpg', temp)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
        cv2.imshow("Template", temp)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    return rect, temp, cv_img
