import numpy as np
import cv2
from __main__ import *
import matplotlib.pyplot as plt
import imutils
import os
import random

from imageFileNames import *

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
prgRun = True



def main(prgRun):
    problem =3

    if problem==1:
        directory = './Bolt2/img'
        # directory = str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows": \n'))

        print("Getting images from " + str(directory))
        imageList = imagefiles(directory)  # get a stack of images

        """process each image individually"""
        for i in range(len(imageList)):
            frameDir = directory + '/' + imageList[i]
            frame = cv2.imread(frameDir)

            cv2.imshow('Working Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    if problem==2:
        directory = './Car4/img'
        # directory = str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows": \n'))

        print("Getting images from " + str(directory))
        imageList = imagefiles(directory)  # get a stack of images

        """process each image individually"""
        for i in range(len(imageList)):
            frameDir = directory + '/' + imageList[i]
            frame = cv2.imread(frameDir)

            cv2.imshow('Working Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    if problem==3:
        directory = './DragonBaby/img'
        # directory = str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows": \n'))

        print("Getting images from " + str(directory))
        imageList = imagefiles(directory)  # get a stack of images

        """process each image individually"""
        for i in range(len(imageList)):
            frameDir = directory + '/' + imageList[i]
            frame = cv2.imread(frameDir)

            cv2.imshow('Working Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


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