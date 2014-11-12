"""Project 3: Tracking.

In this project, you'll track objects in videos.
"""

import cv2
import math
import numpy as np

from tracking_interfaces import *
from tracking_mocks import *
from tracking_attempts import *

DEBUG = False


# computes distance between 2D points
def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


class HoughCirclesBallDetector(ObjectDetector):
    """
    Directly uses the Hough Circles algorithm to detect circles.
    """

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
            minDist=30,
            param1=100,
            param2=50
            )
        # get first element of result
        # get from HoughCircles
        circle = circles[0][0]

        # circle's info
        x = circle[0]
        y = circle[1]
        r = circle[2]

        return (x-r, y-r, x+r, y+r)


class BackgroundSubtractionDetector(ObjectDetector):
    """
    Attempts to subtract the current frame from a mean image of the frames seen
    so far and then use a Hough Circles algorithm to detect a circle in the
    resulting image.
    """

    # Used to ensure that the first frame isn't used in subtraction.
    count = 1
    # Stores averaged background image
    avg = None

    lastCenter = (0, 0)

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

        # Using Gaussian blur to reduce noise and avoid false circle detection
        grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)

        p2 = 45
        circles = []
        while p2 >= 5 and (circles is None or len(circles) < 1):
            circles = cv2.HoughCircles(
                # grayscale image
                grayscale,
                method=cv2.cv.CV_HOUGH_GRADIENT,
                dp=2,
                minDist=300,
                param1=100,
                param2=p2
            )
            p2 -= 5

        cx, cy = self.lastCenter
        r = 0
        if self.count < 4:
            bestCircle = min(
                circles, key=lambda c: dist(c[0, 0], c[0, 1], cx, cy))
        else:
            bestCircle = circles[0]

        self.lastCenter = (bestCircle[0, 0], bestCircle[0, 1])

        for circle in circles:
            for (x, y, r) in circle:
                # circle outline
                cv2.circle(frame, (x, y), r, (0, 0, 255), 3, 8, 0)

        cv2.circle(frame,
                   self.lastCenter,
                   bestCircle[0, 2],
                   (0, 255, 0),
                   4, 8, 0)

        self.count += 1

        x, y, r = bestCircle[0]
        return (x-r, y-r, x+r, y+r)


class AverageRadiusSmoother(Tracker):
    """
    This attempts to prevent large jumps in the size of the ball detected since
    the ball does not actually change size in the frame. Note that this
    approach is not applicable to the face tracking, because the face may
    change size in the frame.
    """

    tracker = None
    avg = 0
    n = 0

    def __init__(self, tracker=None):
        self.tracker = tracker if tracker is not None else NullTracker()

    def observe(self, result):
        radius = self.radius(result)

        self.avg *= float(self.n)
        self.avg += radius
        self.n += 1
        self.avg /= float(self.n)

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
    """
    Uses Haar-cascade Face Detection to find regions of an image that appear to
    contain human faces.
    """

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


def genericTrack(video, detector, tracker, useSecondPass=False):
    """Takes a video object, an ObjectDetector, and a Tracker and returns a
    list of the tracked coordinates for each frame. If the DEBUG flag is true,
    it also outputs frames to the debug_frames directory that show the measured
    and predicted bounding boxes.
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

    if useSecondPass:
        # this is implemented in tracking_attempts.py
        # see the doc comment there and the README for an explanation of what
        # this does.
        print "running second outlier filtering pass..."
        results = filterOutliers(results)

    return results


def findBackground(video):
    """
    Computes a mean image across all the frames of the given video, and returns
    that, as well as the list of frames (to avoid re-reading them from the
    video file).
    """

    ret, frame = video.read()
    frames = []
    avg = np.float32(frame)
    while (ret):
        cv2.accumulateWeighted(frame, avg, 0.1)
        frames.append(frame)
        ret, frame = video.read()
    background = cv2.convertScaleAbs(avg)
    video.release()
    return background, frames


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
    result = []
    lastCircle = []
    THRESHOLD = 7

    fgbg = cv2.BackgroundSubtractorMOG()
    background, frames = findBackground(video)
    fgbg.apply(background)

    for frame in frames:
        fgmask = fgbg.apply(frame)

        _, threshold = cv2.threshold(fgmask, 127, 255, 0)

        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.boundingRect(contours[0])

        if len(lastCircle) != 0 and rect[2] < lastCircle[2] - THRESHOLD:
            result.append((
                lastCircle[0],
                lastCircle[1],
                lastCircle[0] + lastCircle[2],
                lastCircle[1] + lastCircle[3]
            ))
        else:
            result.append((
                rect[0],
                rect[1],
                rect[0] + rect[2],
                rect[1] + rect[3]
            ))
            lastCircle = rect[:4]

    return result


def track_ball_4(video):
    """As track_ball_1, but for ball_2.mov."""

    # List stores (x_min, y_min, x_max, y_max)
    result = []
    # Store the radius of previous tracked circle
    radius = 0
    # Threshold for radius
    threshold = 5
    ret, frame = video.read()

    while ret:
        # convert frame to gray
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Using Gaussian blur to reduce noise
        # and avoid false circle detection
        grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)

        # use loop to make sure p2 goes down to a number
        # that Houghcircles can catch a circle
        p2 = 45
        circles = []
        while p2 >= 5 and (circles is None or len(circles) < 1):
            circles = cv2.HoughCircles(
                # grayscale image
                grayscale,
                method=cv2.cv.CV_HOUGH_GRADIENT,
                dp=3,
                minDist=300,
                param1=100,
                param2=p2
            )
            p2 -= 5

        assert circles is not None

        # Get the first circle
        x, y, r = circles[0][0]

        # Avoid radius of the circle changing too much between frames
        if r < radius - threshold:
            r = radius
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 4)
        res = (x - r, y - r, x + r, y + r)
        result.append(res)
        ret, frame = video.read()

    return result


def track_face(video):
    """As track_ball_1, but for face.mov."""
    return genericTrack(
        video,
        FaceDetector(),
        NullTracker()
    )
