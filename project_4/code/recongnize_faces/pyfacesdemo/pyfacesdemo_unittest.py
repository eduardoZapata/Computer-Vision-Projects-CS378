from pyfaces import pyfaces
import cv2
import sys
import time


def test_image(image, name, dirname):
    img = cv2.imread(image)
    cv2.imshow("cat", img)
    # pyf = pyfaces.PyFaces(img,dirname,4,3)
    # if pyf == name:
    # 	return True
    # else:
    # 	return False


if __name__ == "__main__":

    dirname = sys.argv[1]
    print dirname
    # First unit test - tests if correct name is retured
    image = '../lauren_image.jpg'
    test_image(image, 'cat woman', dirname)
