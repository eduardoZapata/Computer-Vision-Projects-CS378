"""
Mock implementations of the interfaces in tracking_interfaces.py.
These allow us to implement and test detection and tracking separately by
having a dummy implementation of the other for each of them.
"""

from tracking_interfaces import *
import random


class NullTracker(Tracker):
    """A Tracker that just predicts whatever was last observed."""

    def observe(self, answer):
        self.answer = answer

    def predict(self):
        return self.answer


class NoisyGroundTruthDetector(ObjectDetector):
    """
    This is an ObjectDetector that just reads the ground truth answers from
    the file for the videos and produces those numbers with some noise.
    It's not used in what we're turning in, but was written so that we could
    test the tracking independently from the object detection.
    """

    answers = None
    index = 0

    def __init__(self, filename):
        self.answers = []
        with open(filename, "r") as f:
            for line in f:
                answer = map(int, line.split(' '))
                answer = map(lambda v: v + random.randint(-25, 25), answer)
                self.answers.append(answer)

    def detect(self, frame):
        answer = self.answers[self.index]
        self.index += 1
        return answer


class BallNoisyGroundTruthDetector(NoisyGroundTruthDetector):
    def __init__(self):
        NoisyGroundTruthDetector.__init__(self, "test_data/ball_bounds.txt")


class FaceNoisyGroundTruthDetector(NoisyGroundTruthDetector):
    def __init__(self):
        NoisyGroundTruthDetector.__init__(self, "test_data/face_bounds.txt")
