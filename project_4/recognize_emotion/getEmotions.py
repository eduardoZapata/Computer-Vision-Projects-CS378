import cv2
import time
from PIL import Image
import sys,time
from collections import Counter

faceCascade = cv2.CascadeClassifier('pyfacesdemo/haarcascade_frontalface_alt2.xml')
video = cv2.VideoCapture(0)
ret, frame = video.read()
index = 0
start = time.time()
cv2.imshow('Video', frame)
time.sleep(5)
while index < 10:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 5, \
    0 | cv2.cv.CV_HAAR_SCALE_IMAGE, (30, 30))

    # Draw a rectangle around the faces            
    for (x, y, w, h) in faces:
        imgname=gray[y: y + 300, x: x + 300]                               
        im = Image.fromarray(imgname)
        im.save('emotions_db/scared_%d.png' % index)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    cv2.waitKey(1)

    ret, frame = video.read()
    index = index + 1

video.release()
cv2.destroyAllWindows()