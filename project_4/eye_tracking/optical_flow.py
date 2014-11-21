"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""
"""Project 3: Tracking.

Extra credit - multi tracking pedestrians
"""

import cv2
import numpy as np


def multi_tracking(video):
    print "starting"

    ret, frame1 = video.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    prev_objects = []
    tolerance = 50

    framenum = 0; 
    while(1):
        ret, frame2 = video.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        edges = cv2.Canny(rgb, 150, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        #prev_objects = new_prev_objects
        #cv2.imshow('frame2', frame2)
        
        #filename = 'frame_%03d.png'% framenum
        #framenum += 1
        #cv2.imwrite(filename, frame2)
        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        prvs = next

    video.release()
    cv2.destroyAllWindows()



video = cv2.VideoCapture('test_data/IMG_3500.MOV')
multi_tracking(video)
