import cv2
import os
import numpy as np

def createtemplate(frame, rect):
    left = rect[0]
    right = rect[1]
    top = rect[2]
    bottom = rect[3]
    template = frame[top:bottom, left: right]

    # cv2.imshow('template Frame', template)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     None

    # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    return template

def firstTemplateAndSecondFrameAndImageList(path):
    images = []
    cv_img = []
    for image in os.listdir(path):
        images.append(image)
    images.sort()
    for image in images:
        img = cv2.imread("%s/%s" % (path, image))
        cv_img.append(img)
    """get rectangle and template"""
    # rect, tempt = temp.rectAndTemp_problem1()
    return cv_img


def subImageByBoundingBox(img, rect):
    assert type(img) == np.ndarray and type(rect) == np.ndarray
    assert rect.shape == (3, 2)
    # point_upperLeft = rect[0:2, 0].astype(int)
    # point_lowerRight = rect[0:2, 1].astype(int)
    left=int(rect[0,0])
    right=int(rect[0,1])
    top=int(rect[1,1])
    bottom=int(rect[1,0])
    img_cropped = img[top:bottom, left: right]

    cv2.imshow('working Frame', img_cropped)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        None

    # assert img_cropped.shape[1] <= point_lowerRight[0]-point_upperLeft[0], "img cut to size: " + str(img_cropped.shape[0]) + " in x, and supposed to be " + str(int(point_lowerRight[0]-point_upperLeft[0]))    # number of pixles in x smaller than range
    # assert img_cropped.shape[0] <= point_lowerRight[1]-point_upperLeft[1], "img cut to size: " + str(img_cropped.shape[1]) + " in y, and supposed to be " + str(int(point_lowerRight[1]-point_upperLeft[1]))    # number of pixles in y smaller than range
    return img_cropped


def drawRect(img, rect, flag=False):
    p1, p2 = np.transpose(rect.astype(int))[:, 0:2]
    p2 = tuple(p2)
    p1 = tuple(p1)
    img = cv2.rectangle(img, p1, p2, (0, 255, 0), 3)
    if flag:
        cv2.imshow("Image with marked feature", img)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    return img