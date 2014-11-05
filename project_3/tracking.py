"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
from cv2 import cv  # old version of OpenCV required for KalmanFilter
import math
import numpy as np

from tracking_interfaces import *
from tracking_mocks import *

DEBUG = False


# inspired by http://jayrambhia.wordpress.com/2012/07/26/kalman-filter/
class KalmanTracker(Tracker):
    inited = False
    kf = None
    measurement = None
    timestep = 0

    def __init__(self):
        # state vector:
        #  0: min_x
        #  1: min_y
        #  2: max_x
        #  3: max_y
        #  4-7: derivatives of 0-4 respectively
        # measurement vector:
        #  0-4: same as state (no derivatives)
        self.kf = cv.CreateKalman(8, 4, 0)

        self.setOldCvMat(self.kf.transition_matrix, [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])

        self.measurement = cv.CreateMat(4, 1, cv.CV_32FC1)

        cv.SetIdentity(self.kf.measurement_matrix, cv.RealScalar(1))
        cv.SetIdentity(self.kf.process_noise_cov, cv.RealScalar(1e-3))
        cv.SetIdentity(self.kf.measurement_noise_cov, cv.RealScalar(1e-1))
        cv.SetIdentity(self.kf.error_cov_post, cv.RealScalar(1))

    def setOldCvMat(self, cvmat, arr):
        for r, row in enumerate(arr):
            for c, v in enumerate(row):
                cvmat[r, c] = v

    # as suggested at http://dsp.stackexchange.com/questions/3039
    #                 /kalman-filter-implementation-and-deciding-parameters
    # this progressively increases the processNoiseCov parameter. This controls
    # what the Kalman filter perceives as the amount stochastic noise in the
    # model to mitigate the issue of it becoming more trusting of itself and
    # less trusting of new observations over time.
    def updateProcessNoiseCov(self):
        self.timestep += 1

        startTS = 0
        endTS = 400
        startCov = 1e-4
        endCov = 1e-3

        covSlope = (endCov - startCov) / (endTS - startTS)
        covIntercept = startCov

        cov = covSlope * (self.timestep - 30) + covIntercept
        print "frame %d: cov=%.4g" % (self.timestep, cov)

        cv.SetIdentity(self.kf.process_noise_cov, cv.RealScalar(cov))

    def observe(self, measurement):

        measurementMatrix = map(lambda v: [v], measurement)

        if not self.inited:
            self.inited = True

            self.setOldCvMat(self.kf.state_pre, measurementMatrix)

        else:

            self.setOldCvMat(self.measurement, measurementMatrix)
            cv.KalmanCorrect(self.kf, self.measurement)

    def predict(self):

        self.updateProcessNoiseCov()

        prediction = cv.KalmanPredict(self.kf)
        return tuple([int(prediction[i, 0]) for i in range(0, 4)])


def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


class HoughCirclesBallDetector(ObjectDetector):
    # last = None

    # def __init__(self):
    #     self.last = (0, 0, 0, 0)

    lastCenter = (0, 0)

    def detect(self, frame):
        # convert frame to gray
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Using Gaussian blur to reduce noise
        # and avoid false circle detection
        grayscale = cv2.GaussianBlur(grayscale, (5, 5), 2, 2)
        circles = cv2.HoughCircles(
            # grayscale image
            grayscale,
            method=cv2.cv.CV_HOUGH_GRADIENT,
            dp=1,
            minDist=1000,
            param1=100,
            param2=40
            )

        cx, cy = self.lastCenter

        bestCircle = min(circles, key=lambda c: dist(c[0, 0], c[0, 1], cx, cy))

        # circle's info
        x = bestCircle[0, 0]
        y = bestCircle[0, 1]
        self.lastCenter = (x, y)
        r = bestCircle[0, 2]

        if DEBUG:
            cv2.circle (frame, (x, y), r, (0, 0, 255), 3, 8, 0)
            cv2.imshow("image", frame)
            cv2.waitKey(1)

        return (x-r, y-r, x+r, y+r)


class BackgroundSubtractionDetector(ObjectDetector):
    # Used to ensure that the first frame isn't used in subtraction.
    count = 1
    # Stores averaged background image
    avg = None

    lastCenter = (0,0)

    def detect(self, frame):
        if self.avg is None:
            self.avg = np.float32(frame)

        # Add current frame to avg
        cv2.accumulateWeighted(frame, self.avg, 0.01)
        res = cv2.convertScaleAbs(self.avg)

        orig = frame
        frame = cv2.subtract(frame, res)
        if self.count < 2:
            frame = orig

        # convert frame to gray
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Using Gaussian blur to reduce noise and avoid false circle detection
        grayscale = cv2.GaussianBlur (grayscale, (5, 5), 2, 2)

        p2 = 45
        circles = []
        while p2 >= 5 and (circles is None or len(circles) < 1):
            print "trying p2=%d" % p2
            circles = cv2.HoughCircles(
                # grayscale image
                grayscale, 
                method=cv2.cv.CV_HOUGH_GRADIENT, 
                dp=1,
                minDist=300, 
                param1=100,
                param2=p2
            )
            p2 -= 5

        cx, cy = self.lastCenter
        r = 0
        # print circles
        bestCircle = min(circles, key=lambda c: dist(c[0, 0], c[0, 1], cx, cy)) if self.count < 4 else circles[0]
        # print bestCircle
        
        self.lastCenter = (bestCircle[0, 0], bestCircle[0, 1])

        if DEBUG:
            print circles
            for circle in circles:
                for (x,y,r) in circle:
                    # print x
                    # print y
                    # circle outline
                    cv2.circle (frame, (x, y), r, (0, 0, 255), 3, 8, 0)
            
            cv2.circle (frame, self.lastCenter, bestCircle[0, 2], (0, 255, 0), 4, 8, 0)
            
            cv2.imshow("background", res)
            cv2.imshow("image", frame)
            cv2.waitKey(1)

        self.count += 1

        x, y, r = bestCircle[0]
        return (x-r, y-r, x+r, y+r)


class AverageRadiusSmoother(Tracker):
    tracker = None
    avg = 0
    n = 0

    def __init__(self, tracker = None):
        self.tracker = tracker if tracker is not None else NullTracker()

    def observe(self, result):
               
        radius = self.radius(result)
        
        self.avg *= float(self.n)
        self.avg += radius
        self.n += 1
        self.avg /= float(self.n)

        # print "read radius %.4f, avg radius %.4f" % (radius, self.avg)

        adjusted = self.adjustRadius(result, self.avg)

        self.tracker.observe(adjusted)

    def predict(self):
        return self.tracker.predict()

    def radius(self, (min_x, min_y, max_x, max_y)):
        
        w = max_x - min_x
        h = max_y - min_y
        return (w + h) / 4

    def adjustRadius(self, result, radius):

        min_x, min_y, max_x, max_y = result
        
        oldRad = self.radius(result)
        cx, cy = min_x + oldRad, min_y + oldRad

        return (
            int(cx - radius),
            int(cy - radius),
            int(cx + radius),
            int(cy + radius)
        )


class FaceDetector(ObjectDetector):
    # using Haar-cascade Detection
    def detect(self, frame):
        # Load the pre trained classifiers for face
        face_cascade = cv2.CascadeClassifier(
            './haarcascade_frontalface_default.xml')

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect objects
        faces = face_cascade.detectMultiScale(
            # Gray image
            frame_gray,
            # Parameter specifying how much the image
            # size is reduced at each image scale
            scaleFactor=1.133,
            # Parameter specifying how many neighbors
            # each candidate retangle should have to retain it
            minNeighbors=7,
            # It seems using flags gives no different result
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        # convert from (x, y, w, h) to (min_x, min_y, max_x, max_y)
        # using the first result will cause pretty much noise
        # compare to previous position would elimites this noise
        # but don't need to pass the test
        result = faces[0]
        result[2] += result[0]
        result[3] += result[1]

        return result


def genericTrack(video, detector, tracker):
    """Takes a video object, an ObjectDetector, and a Tracker and returns a
    list of the tracked coordinates for each frame
    """

    results = []
    frameNumber = 0

    while True:
        frameNumber += 1
        ret, frame = video.read()
        if not ret:
            break

        observation = detector.detect(frame)
        tracker.observe(observation)
        prediction = tracker.predict()
        results.append(prediction)

        # draw the observation and prediction on the frame and write out to the
        # debug_frames directory
        if DEBUG:
            oColor, pColor = (0, 0, 255), (0, 255, 0)
            cv2.rectangle(
                frame,
                tuple(observation[:2]),
                tuple(observation[2:]),
                oColor,
                3
            )
            cv2.rectangle(
                frame,
                tuple(prediction[:2]),
                tuple(prediction[2:]),
                pColor,
                2
            )
            cv2.imwrite("test_data/debug_frames/%d.png" % frameNumber, frame)

    print "processed %d frames" % len(results)
    return results


def track_ball_1(video):
    """Track the ball's center in 'video'.

    Arguments:
      video: an open cv2.VideoCapture object containing a video of a ball
        to be tracked.

    Outputs:
      a list of (min_x, min_y, max_x, max_y) four-tuples containing the pixel
      coordinates of the rectangular bounding box of the ball in each frame.
    """
    # this uses NullTracker because KalmanTracker is too reluctant and in this
    # test, HoughCirclesBallDetector produces very little noise.
    return genericTrack(
        video,
        HoughCirclesBallDetector(),
        NullTracker()
    )


def track_ball_2(video):
    """As track_ball_1, but for ball_2.mov."""
    return genericTrack(
        video,
        BackgroundSubtractionDetector(),
        AverageRadiusSmoother(NullTracker())
    )


def track_ball_3(video):
    """As track_ball_1, but for ball_2.mov."""
    return genericTrack(
        video,
        BackgroundSubtractionDetector(),
        AverageRadiusSmoother(NullTracker())
    )


def track_ball_4(video):
    """As track_ball_1, but for ball_2.mov."""
    return genericTrack(
        video,
        BackgroundSubtractionDetector(),
        AverageRadiusSmoother()
    )


def track_face(video):
    """As track_ball_1, but for face.mov."""
    return genericTrack(
        video,
        FaceDetector(),
        NullTracker()
    )
