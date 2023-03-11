import cv2 as cv
import numpy as np
import module as m

cameraId = 0

camera = cv.VideoCapture(cameraId)

while(True):
    # getting frame from the camera
    ret, frame = camera.read()

    # converting frame into Gry image.
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calling the face detector funciton
    image, face = m.faceDetector(frame, grayFrame)

    # calling landmarks detector funciton.
    image, PointList = m.faceLandmakDetector(frame, grayFrame, face, False)

    cv.imshow('Frame', image)

    key = cv.waitKey(1)

    if key == ord('q'):
        break

camera.release()

cv.destroyAllWindows()