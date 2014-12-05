import cv2
import numpy as np
import  sys
from string import split
from os.path import basename
from pyfaces import eigenfaces

def main():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    for i in range(0,10):
        frame = cv2.imread("../emotion/neutral_%d.jpg"  % i)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Write image to file
        for (x, y, w, h) in faces:
            scaled_img = scale(frame, (x, y, w, h))
            cv2.imwrite('../emotion/new/neutral_%d.png' % i, scaled_img)

    # Create eigenfaces
    facet = eigenfaces.FaceRec()
    imgnamelist = facet.parsefolder("../emotion/new","png")
    fb = facet.createFaceBundle(imgnamelist)


# takes in tuple of (x, y, w, h)
def scale(source, face):
    crop_img = source[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
    resized_image = cv2.resize(crop_img, (100, 100))
    h, w, _ = resized_image.shape
    resized_image = resized_image[w / 2:w, 0:h]
    print resized_image.shape

    return resized_image


if __name__ == "__main__":
    main()
