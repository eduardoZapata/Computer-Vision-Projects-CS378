#!/usr/bin/env python
import cv2
import numpy as np

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video = cv2.VideoCapture('test_data/Eye Movement Terminology.mp4')

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
    eye_box = {}
    eye_box['x'] = min(i[0] for i in eyes)
    eye_box['y'] = min(i[1] for i in eyes)
    eye_box['w'] = max(i[0] + i[2] for i in eyes)
    eye_box['h'] = max(i[1] + i[3] for i in eyes)

    print eye_box

    cv2.rectangle(frame, (eye_box['x'], eye_box['y']),
                  (eye_box['w'], eye_box['h']), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
