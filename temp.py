import cv2
import numpy as np

def rectAndTemp_problem1(flag=False):
    img1 = cv2.imread('./Car4/img/0001.jpg')
    xl = 65
    xu = 180
    yl = 45
    yu = 135
    rect = np.array([[xl, xu], [yl, yu], [1, 1]])
    temp = img1[yl:yu,xl:xu]
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    if flag:
        cv2.imwrite('./Car4/template.jpg', temp)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
    return rect, temp