#!/usr/bin/env python
import cv2
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    exit()

video = cv2.VideoCapture(sys.argv[1])

frame_count = 0

while True:
    ret, frame = video.read()

    if frame is None:
        break

    if frame_count % 45 == 0:
        print frame_count
        plt.imshow(frame)
        plt.show()

    frame_count += 1

print frame_count


