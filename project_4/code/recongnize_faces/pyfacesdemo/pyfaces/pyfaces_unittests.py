from pyfaces import pyFaces
import cv2
import time
from PIL import Image
import sys,time

def test_image(image, name, dirname):
	img = cv2.imread(image)
	pyf = pyfaces.PyFaces(image,dirname,4,3)


if __name__ == '__main__':
	dirname = sys.argv[1]

	#First unit test for Facial Recognition
	image = '../lauren_image.jpg'
	test_image(image, 'cat woman', dirname)