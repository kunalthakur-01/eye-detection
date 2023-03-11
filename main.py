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

    RightEyePoint = PointList[36:42]
    LeftEyePoint = PointList[42:48]

    if(len(RightEyePoint) and len(LeftEyePoint)):
        leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
        rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)
        cv.circle(image, topMid, 2, m.YELLOW, -1)
        cv.circle(image, bottomMid, 2, m.YELLOW, -1)


    for p in LeftEyePoint:
        cv.circle(image, p, 3, m.MAGENTA, 1)

    cv.imshow('Frame', image)

    key = cv.waitKey(1)

    if key == ord('q'):
        break

camera.release()

cv.destroyAllWindows()