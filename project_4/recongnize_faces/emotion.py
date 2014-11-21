import cv2
import numpy as np

def main():
    img = cv2.imread("faces_db/bryan_2.png")
    hist = histogram_equalize(img)

    cv2.imshow("orig", img)
    cv2.imshow("hist", hist)

    cv2.imshow("affine", scale())

    cv2.waitKey(4000)


# Scale image according to both eyes
# takes in tuple for each eye and one for mouth
def scale(): # eye1, eye2, mouth
    img = cv2.imread("faces_db/bryan_2.png")
    mask = ([(50,110), (93,34), (114,103)])
    mask2 = ([(60,100),(90,30),(120,100)])

    mask = np.array(mask)
    mask2 = np.array(mask2)

    print mask
    print mask2

    mat = cv2.getAffineTransform(mask, mask2)
    dst = cv2.warpAffine(img, mat, (170, 130))

    return dst


def histogram_equalize(source):
    # Convert it to grayscale
    grey = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization with the function equalizeHist
    hist = cv2.equalizeHist(grey)

    return hist


# Apply mask to extract facial features
def apply_mask(source):
    return

if __name__ == "__main__":
    main()
