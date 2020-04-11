from __main__ import *


def createtemplate(frame, rect):
    left = rect[0]
    right = rect[1]
    top = rect[2]
    bottom = rect[3]
    template = frame[top:bottom,left: right]

    # cv2.imshow('template Frame', template)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     None

    # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    return template


def affineLKtracker(frame,temp,rect,Pprev):


    # Find center

    LX = frame.shape[0]
    LY = frame.shape[1]
    CY = int(LX / 2)
    CX = int(LY / 2)

    NormF1 = frame / 255
    NormFP = frame/ 255



    return CX, CY,Pprev
################################################
