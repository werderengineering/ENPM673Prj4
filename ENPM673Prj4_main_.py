import numpy as np
import cv2
from __main__ import *
import matplotlib.pyplot as plt
import imutils
import os
import random
from scipy import signal

import featureObjectTracking as ft
import imagesProcessing as ip
from temp import *

from imageFileNames import *

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
prgRun = True
flag_trackingFeature = True



def runVidOp(directory, rect):

    tempt,rect,rectT=createtemplate(directory, rect)
    frames = ip.firstTemplateAndSecondFrameAndImageList(
        directory)  # get first bounding box around template, template, frames

    p = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).reshape(6, 1)
    w=np.eye(3)
    cache=[]
    print("Import images done")
    if flag:
        ip.drawRect(frames[0], rect, True)
    for frame in frames:
        # try:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        rect, p, w, cache= ft.affineLKtracker(frame_gray, rect, tempt, rectT, p, w,cache)  # rect update
        tempt = ip.subImageInBoundingBoxAndEq(frame_gray, rect)  # template update
        """show results"""
        frame_featureMarked = ip.drawRect(frame, rect, False)
        if flag_trackingFeature:
            cv2.imshow("Tracking feature", frame_featureMarked)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # except:
        #     print(rect)
        #     pass




def main(prgRun):
    # start file
    problem = 3

    if problem == 1:
        directory = './Bolt2/img'
        # directory = str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows": \n'))
        # left = 250, right = 320, top = 75, bottom = 150

        rect=[250,320,75,150]
        runVidOp(directory,rect)
        print("Problem 1 finished")
        # runVid(directory, rect)

    if problem == 2:
        directory = './Car4/img'
        # directory = str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows": \n'))
        # left = 65, right = 180, top = 45, bottom = 135
        rect = [67, 165, 50, 130]
        runVidOp(directory, rect)
        print("Problem 2 finished")
        # runVid(directory, rect)

    if problem == 3:
        directory = './DragonBaby/img'
        # directory = str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows": \n'))
        # left = 130, right = 220, top = 75, bottom = 300
        rect = [150,220, 75, 150]
        runVidOp(directory, rect)
        print("Problem 3 finished")
        # runVid(directory, rect)

    prgRun = False
    return prgRun


print('Function Initializations complete')

if __name__ == '__main__':
    print('Start')
    prgRun = True
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()
