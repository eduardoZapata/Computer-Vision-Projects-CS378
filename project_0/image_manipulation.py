"""Project 0: Image Manipulation with OpenCV.

In this assignment, you will implement a few basic image
manipulation tasks using the OpenCV library.

Use the unit tests is image_manipulation_test.py to guide
your implementation, adding functions as needed until all
unit tests pass.
"""
import cv2
import numpy as np


def flip_image(image, horizontal, vertical):
    if horizontal and vertical:
        cv2.flip(image, -1, image)
    elif horizontal:
        cv2.flip(image, 1, image)
    elif vertical:
        cv2.flip(image, 0, image)
    return image


def negate_image(image):
    image = cv2.bitwise_not(image)
    return image


def swap_blue_and_green(image):
    b, g, r = cv2.split(image)
    image = cv2.merge((g, b, r))
    return image
