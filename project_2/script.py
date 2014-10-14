import cv2
import numpy as np
import sys
import math
from stereo import *

if __name__ == '__main__':
    print "Reading images..."
    image_left = cv2.imread("my_stereo/img1.png")
    image_right = cv2.imread("my_stereo/img2.png")

    _, h_left, h_right = rectify_pair(image_left, image_right)

    left = cv2.warpPerspective(
        image_left, h_left, (image_left.shape[1], image_left.shape[0]))

    right = cv2.warpPerspective(
        image_right, h_right, (image_right.shape[1], image_right.shape[0]))

    disparity = disparity_map(left, right)
    cv2.imwrite("my_stereo/disparity.png", disparity)

    print "Wrote disparity.png"

    ply_string = point_cloud(disparity, image_left, 4)

    with open("my_stereo/test.ply", "w") as f:
        f.write(ply_string)

    print "Wrote test.ply"
