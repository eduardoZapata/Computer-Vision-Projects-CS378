import cv2
import numpy as np
import sys
from pano_stitcher import *

# x = width = number of columns
# y = height = number of rows
# img.shape returns h, w, channels


if __name__ == '__main__':
    print "Reading images for bookshelf..."
    read_alpha = -1
    images = [cv2.imread("test_data/books_1.png", read_alpha),
              cv2.cvtColor(cv2.imread("test_data/books_2.png"),
                           cv2.COLOR_BGR2BGRA),
              cv2.imread("test_data/books_3.png", read_alpha)]
    result = images[0]

    image_0_warped, pos0 = warp_image(images[0], homography(images[1],
                                      images[0]))
    image_2_warped, pos2 = warp_image(images[2], homography(images[1],
                                      images[2]))
    images = [image_0_warped, image_2_warped, images[1]]
    origins = [pos0, pos2, (0, 0)]
    print pos0
    print pos2

    result = create_mosaic(images, origins)
    cv2.imwrite("test_data/bookshelf.png", result)
    print "Wrote bookshelf.png"

    print "Reading images for Gates-Vertical..."
    read_alpha = -1
    images = [cv2.imread("my_panos/vGates-1.jpg", read_alpha),
              cv2.cvtColor(cv2.imread("my_panos/vGates-2.jpg"),
                           cv2.COLOR_BGR2BGRA),
              cv2.imread("my_panos/vGates-3.jpg", read_alpha)]
    result = images[0]

    image_0_warped, pos0 = warp_image(images[0], homography(images[1],
                                      images[0]))
    image_2_warped, pos2 = warp_image(images[2], homography(images[1],
                                      images[2]))
    images = [image_0_warped, image_2_warped, images[1]]
    origins = [pos0, pos2, (0, 0)]
    print pos0
    print pos2

    result = create_mosaic(images, origins)
    cv2.imwrite("my_panos/Gates-Vertical.png", result)
    print "Wrote Gates-Vertical.png"

    print "Reading images for Gates-Horizontal..."
    read_alpha = -1
    images = [cv2.imread("my_panos/hGates-1.jpg", read_alpha),
              cv2.cvtColor(cv2.imread("my_panos/hGates-2.jpg"),
                           cv2.COLOR_BGR2BGRA),
              cv2.imread("my_panos/hGates-3.jpg", read_alpha)]
    result = images[0]

    image_0_warped, pos0 = warp_image(images[0], homography(images[1],
                                      images[0]))
    image_2_warped, pos2 = warp_image(images[2], homography(images[1],
                                      images[2]))
    images = [image_0_warped, image_2_warped, images[1]]
    origins = [pos0, pos2, (0, 0)]
    print pos0
    print pos2

    result = create_mosaic(images, origins)
    cv2.imwrite("my_panos/Gates-Horizontal.png", result)
    print "Wrote Gates-Horizontal.png"

    print "Reading images for Gates-Inside..."
    read_alpha = -1
    images = [cv2.imread("my_panos/inside-1.jpg", read_alpha),
              cv2.cvtColor(cv2.imread("my_panos/inside-2.jpg"),
                           cv2.COLOR_BGR2BGRA),
              cv2.imread("my_panos/inside-3.jpg", read_alpha)]
    result = images[0]

    image_0_warped, pos0 = warp_image(images[0], homography(images[1],
                                      images[0]))
    image_2_warped, pos2 = warp_image(images[2], homography(images[1],
                                      images[2]))
    images = [image_0_warped, image_2_warped, images[1]]
    origins = [pos0, pos2, (0, 0)]
    print pos0
    print pos2

    result = create_mosaic(images, origins)
    cv2.imwrite("my_panos/Gates-Inside.png", result)
    print "Wrote Gates-Inside.png"
