"""Project 2: Stereo vision.

In this project, you'll extract dense 3D information from stereo image pairs.
"""

import cv2
import math
import numpy


def rectify_pair(image_left, image_right, viz=False):
    """Computes the pair's fundamental matrix and rectifying homographies.

    Arguments:
      image_left, image_right: 3-channel images making up a stereo pair.

    Returns:
      F: the fundamental matrix relating epipolar geometry between the pair.
      H_left, H_right: homographies that warp the left and right image so
        their epipolar lines are corresponding rows.
    """
    # use SIFT with default parameters
    # don't need to change the default values
    sift = cv2.SIFT()

    # Feature Matching using sift
    kp1, des1 = sift.detectAndCompute(image_left, None)
    kp2, des2 = sift.detectAndCompute(image_right, None)

    # FLANN based Matcher
    FLANN_INDEX_KDTREE = 0

    # two dicts need for FLANN based matcher (for SIFT)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    # number of times the trees in the index
    # should be recursively traversed
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance / n.distance < .7:
            good.append(m)

    # change to floating point and reshape
    # to use in findFundamentalMat
    left_points = numpy.float32([kp1[m.queryIdx].pt for m in good])
    left_points = left_points.reshape(-1, 1, 2)
    right_points = numpy.float32([kp2[m.trainIdx].pt for m in good])
    right_points = right_points.reshape(-1, 1, 2)

    # Fundamental matrix - use some default parameters
    f_matrix, mask = cv2.findFundamentalMat(left_points, right_points)

    # Rectify images
    shape = (image_left.shape[0], image_left.shape[1])
    retval, h_left, h_right = cv2.stereoRectifyUncalibrated(
        left_points, right_points, f_matrix, shape)

    return f_matrix, h_left, h_right


def disparity_map(image_left, image_right):
    """Compute the disparity images for image_left and image_right.

    Arguments:
      image_left, image_right: rectified stereo image pair.

    Returns:
      an single-channel image containing disparities in pixels,
        with respect to image_left's input pixels.
    """

    # create SGBM with default parameters
    sgbm = cv2.StereoSGBM()

    # Change to appropriate parameters
    # These parameters pass the unit test

    # Matched block size >= 1 in between 3,11
    sgbm.SADWindowSize = 5
    # Maximum disparity minus minimum disparity % 16 = 0
    sgbm.numberOfDisparities = 192
    # Truncation value for the prefiltered image pixels
    sgbm.preFilterCap = 4
    # Minimum possible disparity value
    sgbm.minDisparity = 10
    # Margin in percentage good in between 5,15
    sgbm.uniquenessRatio = 5
    # Maximum size of smooth disparity regions
    sgbm.speckleWindowSize = 200
    # aximum disparity variation within each connected component
    sgbm.speckleRange = 2
    # Maximum allowed difference
    sgbm.disp12MaxDiff = 10
    # full-scale two-pass dynamic programming algorithm (no)
    sgbm.fullDP = False
    # The first parameter controlling the disparity smoothness
    sgbm.P1 = 600
    # he second parameter controlling the disparity smoothness
    sgbm.P2 = 2400

    # find the disparity
    disparity = sgbm.compute(image_left, image_right)

    # Since Disparity will be either CV_16S or CV_32F
    # it needs to be compressed and normalized to CV_8U
    disparity_8bit = cv2.normalize(
        disparity, alpha=0, beta=255,
        norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8U)

    return disparity_8bit


def point_cloud(disparity_image, image_left, focal_length):
    """Create a point cloud from a disparity image and a focal length.

    Arguments:
      disparity_image: disparities in pixels.
      image_left: BGR-format left stereo image, to color the points.
      focal_length: the focal length of the stereo camera, in pixels.

    Returns:
      A string containing a PLY point cloud of the 3D locations of the
        pixels, with colors sampled from left_image. You may filter low-
        disparity pixels or noise pixels if you choose.
    """
    # wrote code according to stereo_match.py file professor suggest.

    # Get height and width of image_left
    height, width = image_left.shape[:2]

    # projection matrix professor suggests
    # Example
    # [ 1  0             0   image_width / 2 ]
    # [ 0  1             0  image_height / 2 ]
    # [ 0  0  focal_length                 0 ]
    # [ 0  0             0                 1 ]

    # In our case:
    # change some parameters to orient point cloud to correct orientation.
    pMatrix = numpy.float32([[1,  0,             0,   width / 2],
                             [0, -1,             0,  height / 2],
                             [0,  0,  focal_length, -1],
                             [0,  0,             0,           1]])
    # Reprojecting the disparity image to 3D space.
    points = cv2.reprojectImageTo3D(disparity_image, pMatrix)

    # convert image_left to RGB color space (to match with the PLY's format)
    colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)

    # filter low-disparity pixels
    mask = disparity_image > disparity_image.min()

    # get points according to low-disparity pixels
    out_points = points[mask]
    out_colors = colors[mask]
    fileName = 'out.ply'

    # reshape to have 3 elements in a row
    out_points = out_points.reshape(-1, 3)
    out_colors = out_colors.reshape(-1, 3)

    # using numpy.hstack
    # to stack arrays in sequence horizontally (column wise).
    vertices = numpy.hstack([out_points, out_colors])

    # save to string variable in the PLY file format
    # the header of .ply file
    ply_header = '''ply
                format ascii 1.0
                element vertex %(vert_num)d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
                '''
    # add number of vertices to header file
    output = ply_header % dict(vert_num=len(vertices))

    # convert out_points array to string (follow .ply format)

    # organize to "<x> <y> <z> <r> <g> <b>" format
    str_list = [tuple("%4.6f %4.6f %4.6f %i %i %i" % (
        sublist[0], sublist[1], sublist[2],
        sublist[3], sublist[4], sublist[5]))
        for sublist in vertices]
    # add to output string
    output += '\n'.join(''.join(element) for element in str_list)

    # a new line at the end
    # (Sometimes the .ply file can't opened by Meshlab
    # because of missing a new line at the end)
    # add to make sure everything works well
    output += '\n'

    return output
