"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

TODO: Implement!
"""

import cv2
import numpy


def homography(image_a, image_b):
    """Returns the homography mapping image_b into alignment with image_a.

    Arguments:
      image_a: A grayscale input image.
      image_b: A second input image that overlaps with image_a.

    Returns: the 3x3 perspective transformation matrix (aka homography)
             mapping points in image_b to corresponding points in image_a.
    """

    sift = cv2.SIFT()

    kpA, descriptorsA = sift.detectAndCompute(image_b, None)
    kpB, descriptorsB = sift.detectAndCompute(image_a, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptorsA, descriptorsB, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.735 * n.distance:
            good.append(m)

    srcP = numpy.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstP = numpy.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(srcP, dstP, cv2.RANSAC, 1.0)

    return homography


def warp_image(image, homography):
    """Warps 'image' by 'homography'

    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.

    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rows, cols, shape = result.shape

    co = numpy.array([numpy.array([[0, 0], [0, rows], [cols, 0], [cols, rows]],
                     dtype="float32")])

    newCo = cv2.perspectiveTransform(co, homography)

    minRow, minCol = newCo[0][0][1], newCo[0][0][0]
    maxRow, maxCol = minRow, minCol

    for index in xrange(4):
        minRow = min(minRow, newCo[0][index][1])
        minCol = min(minCol, newCo[0][index][0])
        maxRow = max(maxRow, newCo[0][index][1])
        maxCol = max(maxCol, newCo[0][index][0])

    newRows = int(maxRow - minRow)
    newCols = int(maxCol - minCol)

    translation = numpy.matrix([[1, 0, -minCol], [0, 1, -minRow],
                               [0, 0, 1]])

    result = cv2.warpPerspective(result, translation * homography,
                                 (newCols, newRows))

    return (result, (minCol, minRow))


def create_mosaic(images, origins):
    """Combine multiple images into a mosaic.

    Arguments:
      images: a list of 4-channel images to combine in the mosaic.
      origins: a list of the locations upper-left corner of each image in
               a common frame, e.g. the frame of a central image.

    Returns: a new 4-channel mosaic combining all of the input images. pixels
             in the mosaic not covered by any input image should have their
             alpha channel set to zero.
    """
    minX, minY = (origins[0][0], origins[0][1])
    maxX, maxY = minX, minY

    for index in xrange(len(images)):
        image = images[index]
        origin = origins[index]

        rows, cols, shape = image.shape
        x, y = origin

        minX = min(minX, x)
        minY = min(minY, y)

        maxX = max(maxX, x + cols)
        maxY = max(maxY, y + rows)

    result = numpy.zeros(((maxY - minY + 1), (maxX - minX + 1), 4),
                         dtype=numpy.uint8)

    for index in xrange(len(images)):
        image = images[index]
        origin = origins[index]

        rows, cols, shape = image.shape
        x, y = origin

        for r in xrange(rows):
            for c in xrange(cols):
                if image[r][c][3] == 255:
                    result[(y - minY) + r][(x - minX) + c] = image[r][c]

    return result
