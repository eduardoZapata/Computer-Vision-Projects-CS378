"""
General superclasses for object detection and tracking tasks
"""


class Tracker:

    def observe(self, (min_x, min_y, max_x, max_y)):
        """Updates the model with a new observation from a single timestep."""
        pass

    def predict(self):
        """Returns a prediction tuple at the current timestep."""
        return (0, 0, 0, 0)


class ObjectDetector:

    def detect(self, frame):
        """Returns a (min_x, min_y, max_x, max_y) observation tuple for a given
        frame.
        """
        return (0, 0, 0, 0)
