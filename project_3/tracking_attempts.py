
import numpy as np
import cv2
from cv2 import cv
from tracking_interfaces import *
from tracking_mocks import *
from collections import deque


# inspired by http://jayrambhia.wordpress.com/2012/07/26/kalman-filter/
class KalmanTracker(Tracker):
    inited = False
    kf = None
    measurement = None
    timestep = 0

    def __init__(self, useProgressivePNC=False, pnc=1e-3):
        # state vector:
        #  0: min_x
        #  1: min_y
        #  2: max_x
        #  3: max_y
        #  4-7: derivatives of 0-4 respectively
        # measurement vector:
        #  0-4: same as state (no derivatives)
        self.useProgressivePNC = useProgressivePNC
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
        cv.SetIdentity(self.kf.process_noise_cov, cv.RealScalar(pnc))
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
        if not self.useProgressivePNC:
            return

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


class JumpRejector(Tracker):
    """
    This class is a Tracker that attempts to filter out frames for which the
    predicted position jumps by computing a Z score for the X and Y coordinates
    of the center of the last few predictions and rejecting frames with a
    Z-score of excessive magnitude.
    """

    last = None
    tracker = None

    def __init__(self, tracker=None):
        self.tracker = tracker if tracker is not None else NullTracker()
        self.last = deque()

    def observe(self, result):
        self.last.append(result)
        if len(self.last) > 10:
            self.last.popleft()

        def center(box):
            return (box[0] + (box[2] - box[0])/2.0,
                    box[1] + (box[3] - box[1])/2.0)

        points = map(center, self.last)
        meanX = sum((p[0] for p in points)) / float(len(self.last))
        meanY = sum((p[1] for p in points)) / float(len(self.last))
        stdX, stdY = np.std(points, axis=0)

        def isOutlier(point):
            if len(self.last) < 5:
                return False

            x, y = point
            zx, zy = (x - meanX) / stdX, (y - meanY) / stdY

            threshold = 1.5
            return abs(zx) > threshold or abs(zy) > threshold

        for i in range(len(self.last)-1, -1, -1):
            if not isOutlier(center(self.last[i])):
                self.tracker.observe(self.last[i])
                break

    def predict(self):
        return self.tracker.predict()


def filterOutliers(results):
    """
    This is called from the top level genericTrack function and looks at the
    full set of results after all frames have been processed to filter out bad
    results for frames that differ greatly from the frames before or after
    them.
    """

    nr = results[:]

    for i in range(1, len(results)-1):
        last, mid, next = results[i-1:i+2]

        def sqdiff(a, b):
            return sum(((i-j)**2 for i, j in zip(a, b)))

        diff = sqdiff(last, mid) + sqdiff(mid, next)

        if diff > 6500:
            nr[i] = last

    return nr
