#!/usr/bin/env python
import cv2
import cv2.cv as cv
import numpy as np

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# video = cv2.VideoCapture('test_data/Eye Movement Terminology.mp4')
video = cv2.VideoCapture('test_data/IMG_3500.MOV')

while True:
    ret, frame = video.read()

    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    """ We'll need to consider whether we want to use do face tracking
    here as well. I don't, just because the video is zoomed in too close
    to recognize a face. It is something important to consider, though
    """
    eyes = eye_cascade.detectMultiScale(gray_frame)

    """ eye_box is our ROI. It's a rectangle that encompasses both eyes,
    because (for most people), the eyes don't move independently, and thus
    there is no reason to anylysis the two eyes seperately
    """
    # eye_box = {}
    # eye_box['x'] = min(i[0] for i in eyes)
    # eye_box['y'] = min(i[1] for i in eyes)
    # eye_box['w'] = max(i[0] + i[2] for i in eyes)
    # eye_box['h'] = max(i[1] + i[3] for i in eyes)

    # eye_img = gray_frame[eye_box['y']:eye_box['h'], eye_box['x']:eye_box['w']]

    # eye_img_blurred = cv2.medianBlur(eye_img, 5)
    # Find the circle based on the hough circle transform
    # circles = cv2.HoughCircles(eye_img_blurred, cv2.cv.CV_HOUGH_GRADIENT, 1,
    #                            70, param1=40, param2=31, minRadius=0, maxRadius=0)

    for (ex, ey, ew, eh) in eyes:
        eye_frame = gray_frame[ey:ey+eh, ex:ex+ew]
        eye_blurred = cv2.medianBlur(eye_frame, 5)

        circles = cv2.HoughCircles(eye_blurred, cv2.cv.CV_HOUGH_GRADIENT, 1,
                                   70, param1=40, param2=31, minRadius=0, maxRadius=0)

        if circles is None:
            continue

        circles = np.uint16(np.around(circles))
        cimg = eye_blurred.copy()

        index = 0
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

            cv2.imshow('circles', cimg)

    cv2.waitKey(0)

    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     cimg = eye_img_blurred.copy()

    #     index = 0
    #     for i in circles[0,:]:
    #         # draw the outer circle
    #         cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #         # draw the center of the circle
    #         cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    #     cv2.imshow('circles', cimg)
    #     cv2.waitKey(0)
                
    # print eye_box


    # for (ex, ey, ew, eh) in eyes:
    #     cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)

    # cv2.rectangle(frame, (eye_box['x'], eye_box['y']),
    #               (eye_box['w'], eye_box['h']), (0, 255, 0), 2)

    # cv2.imshow('frame', frame)

