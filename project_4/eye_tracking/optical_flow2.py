"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""
"""Project 3: Tracking.

Extra credit - multi tracking pedestrians
"""

import cv2
import numpy as np


def findBackground(frames):
    """ Reads n number of frames and uses a weighted average
        to estimate the background of the video.

        Arguments:
          frames: a list of consecutive image frames from a video

        Outputs:
          A single image representing the estimated background of the video
    """

    n = 298  # Number of frames to use
    i = 1
    background = frames[0]
    # Read each frame and do a weighted sum
    # beta is the weight for each new frame which starts at 1/2 and
    # gets smaller
    # alpha is the weight of the sum and starts at 1/2 and gets larger
    while(i < n):
        i += 1
        frame = frames[i]
        beta = 1.0 / (i + 1)
        alpha = 1.0 - beta
        background = cv2.addWeighted(background, alpha, frame, beta, 0.0)

    # Comment these lines back in to see result
    cv2.imshow('estimated background', background)
    cv2.waitKey(3000)

    return background


def readFrames(video):
    """ Simply reads all the frames from video and
        stores them in a list which is returned.
        We do this so that we can read some number
        of frames to use in background estimation
        without having to reset the video capture.

        Arguments:
          video: an open cv2.VideoCapture object

        Outputs:
          A list of all the frames in consecutive order from the video
    """
    cv2.namedWindow("input")
    frames = []
    i = 0
    max = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    print "max ", max
    while i < 1000:
        _, frame = video.read()
        if frame is None:
            print "here"
        else:
            frames.append(frame)
        i += 1
    video.release()
    return frames


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

        flow = cv2.calcOpticalFlowFarneback(prvs, next,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        edges = cv2.Canny(rgb, 150, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        """
        new_prev_objects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            found_obj = False

            if w*h > 400:
                for new_prev_obj in new_prev_objects:
                    if (abs(x - new_prev_obj[0]) < tolerance and
                            abs(y - new_prev_obj[1]) < tolerance):
                        found_obj = True
                        break
                if not found_obj:
                    new_prev_objects.append((x, y))

            contains = False
            if not found_obj:
                for prev_object in prev_objects:
                    if (abs(x - prev_object[0]) < tolerance and
                            abs(y - prev_object[1]) < tolerance):
                        if w*h > 400:
                            cv2.rectangle(frame2,
                                          (x, y),
                                          (x + w, y + h),
                                          (0, 255, 0), 2)
        """
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
