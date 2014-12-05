#!/usr/bin/env python
import cv2
import cv2.cv as cv
import numpy as np

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
video = cv2.VideoCapture('test_data/eyes.mp4')
rightEye = []
index = 0

while True:

    ret, frame = video.read()

    # if no frame then break loop and exit program
    if not ret:
        break

    # Convert to gray frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    """ We'll need to consider whether we want to use do face tracking
    here as well. I don't, just because the video is zoomed in too close
    to recognize a face. It is something important to consider, though
    """
    eyes = eye_cascade.detectMultiScale(gray_frame)
    """ eye_box is our ROI. It's a rectangle that encompasses both eyes,
    because (for most people), the eyes don't move independently, and thus
    there is no reason to analyze the two eyes separately
    """
    eye_box = {}
    eye_box['x'] = min(i[0] for i in eyes)
    eye_box['y'] = min(i[1] for i in eyes)
    eye_box['w'] = max(i[0] + i[2] for i in eyes)
    eye_box['h'] = max(i[1] + i[3] for i in eyes)

    eye_frame = gray_frame[eye_box['y']:eye_box['h'],
        eye_box['x']:eye_box['w']]

    eye_frame_color = frame[eye_box['y']:eye_box['h'],
        eye_box['x']:eye_box['w']]

    eye_blurred = cv2.medianBlur(eye_frame, 5)

    circles = cv2.HoughCircles(eye_blurred, cv2.cv.CV_HOUGH_GRADIENT, 1,
    100, param1=40, param2=30, minRadius=0, maxRadius=20)

    if circles is None:
        continue

    circles = np.uint16(np.around(circles))
    middle = eye_frame_color.shape[1] / 2

    for i in circles[0, :]:

        # draw the center of the circle
        cv2.circle(eye_frame_color, (i[0], i[1]), 2, (0, 255, 0), 2)
        cv2.imshow('circles', frame)

        if(i[0] < middle):
            rightEye.append(i)
            index = index + 1
        else:
            break

        if index > 7:
            previousEyeH = rightEye[index - 7]

            previousEyeX = previousEyeH[0]
            previousEyeY = previousEyeH[1]

            currentEye = i
            currentEyeX = i[0]
            currentEyeY = i[1]

            if previousEyeY > currentEyeY:
                print ' moving up '
            if previousEyeY < currentEyeY:
                print ' moving down '
            if previousEyeX < currentEyeX:
                print ' moving right '
            if previousEyeX > currentEyeX:
                print ' moving left '
            if previousEyeX == currentEyeX:
                print ' not moving '

    cv2.waitKey(100)
