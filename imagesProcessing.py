import cv2
import os
import numpy as np
import temp

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
    rect, tempt = temp.rectAndTemp_problem1()
    return rect, tempt, cv_img


def subImageByBoundingBox(img, rect):
    assert type(img) == np.ndarray and type(rect) == np.ndarray
    assert rect.shape == (3, 2)
    point_upperLeft = rect[0:2, 0].astype(int)
    point_lowerRight = rect[0:2, 1].astype(int)
    img_cropped = img[point_upperLeft[1]:point_lowerRight[1], point_upperLeft[0]:point_lowerRight[0]]
    assert img_cropped.shape[1] <= point_lowerRight[0]-point_upperLeft[0], "img cut to size: " + str(img_cropped.shape[0]) + " in x, and supposed to be " + str(int(point_lowerRight[0]-point_upperLeft[0]))    # number of pixles in x smaller than range
    assert img_cropped.shape[0] <= point_lowerRight[1]-point_upperLeft[1], "img cut to size: " + str(img_cropped.shape[1]) + " in y, and supposed to be " + str(int(point_lowerRight[1]-point_upperLeft[1]))    # number of pixles in y smaller than range
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