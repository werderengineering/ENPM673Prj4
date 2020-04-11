import numpy as np
import cv2
from __main__ import *
import matplotlib.pyplot as plt
import imutils
import os
import random
from scipy import signal

from imageFileNames import *
from OpticalFlow import *

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
prgRun = True

def runVid(directory,rect):
    Pprev=1
    print("Getting images from " + str(directory))
    imageList = imagefiles(directory)  # get a stack of images

    """process each image individually"""
    for i in range(len(imageList)):
        frameDir = directory + '/' + imageList[i]
        frame = cv2.imread(frameDir)

        if i == 0:
            template = createtemplate(frame, rect)

        if i > 1:
            PframeDir = directory + '/' + imageList[i - 1]
            Pframe = cv2.imread(PframeDir)

            framG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            templateG = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            Tcx, Tcy,Pprev = affineLKtracker(framG, templateG, rect, Pprev)

            cv2.circle(frame, (Tcx, Tcy), 10, (0, 0, 255), 2)

            cv2.imshow('Original Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


def main(prgRun):
    # start file
    problem = 2

    if problem == 1:
        directory = './Bolt2/img'
        # directory = str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows": \n'))
        # left = 250, right = 320, top = 75, bottom = 150
        rect = [250, 320, 75, 150]
        runVid(directory, rect)


    if problem == 2:
        directory = './Car4/img'
        # directory = str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows": \n'))
        # left = 65, right = 180, top = 45, bottom = 135
        rect = [65, 180, 45, 135]
        runVid(directory, rect)


    if problem == 3:
        directory = './DragonBaby/img'
        # directory = str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows": \n'))
        # left = 130, right = 220, top = 75, bottom = 300
        rect = [130, 220, 75, 300]

        runVid(directory, rect)


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
