import cv2
import numpy as np
import imagesProcessing as ip

def rectAndTemp_problem1(flag=False):
    frame = cv2.imread('./Bolt2/img/0001.jpg')
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    xl = 250
    xu = 320
    yl = 75
    yu = 150
    rect = np.array([[xl, xu], [yl, yu], [1, 1]])
    # temp = img[yl:yu, xl:xu]
    temp = ip.subImageByBoundingBox(img, rect)
    if flag:
        cv2.imwrite('./Bolt2/template.jpg', temp)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    return rect, temp

def rectAndTemp_problem2(flag=False):
    frame = cv2.imread('./Car4/img/0001.jpg')
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    xl = 65
    xu = 180
    yl = 45
    yu = 135
    rect = np.array([[xl, xu], [yl, yu], [1, 1]])
    temp = ip.subImageByBoundingBox(img, rect)
    if flag:
        cv2.imwrite('./Car4/template.jpg', temp)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    return rect, temp


def rectAndTemp_problem3(flag=False):
    frame = cv2.imread('./DragonBaby/img/0001.jpg')
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    xl = 130
    xu = 220
    yl = 75
    yu = 300
    rect = np.array([[xl, xu], [yl, yu], [1, 1]])
    temp = ip.subImageByBoundingBox(img, rect)
    if flag:
        cv2.imwrite('./DragonBaby/template.jpg', temp)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    return rect, temp


