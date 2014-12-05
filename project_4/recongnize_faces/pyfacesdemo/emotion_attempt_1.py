import cv2
import numpy as np

def main(): 
    img = cv2.imread("emotion_test.png")
    scaled_img = scale(img, (481, 179, 313, 313))
    hist = histogram_equalize(scaled_img)
    masked_img = apply_mask(hist)

    filters = create_filters()
    gabor = process(masked_img, filters)
    pca(gabor)

    # gabor = cv2.bitwise_not(gabor)
    # gabor = apply_mask(gabor)


    # cv2.imshow("orig", img)
    # cv2.imshow("scale", scaled_img)
    # cv2.imshow("hist", hist)
    # cv2.imshow("masked", masked_img)
    # cv2.imshow("gabor", gabor)


# takes in tuple of (x, y, w, h)
def scale(source, face):
    crop_img = source[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
    resized_image = cv2.resize(crop_img, (100, 100)) 

    return resized_image

# Normalizes image to make it invariant to extraneous features
def histogram_equalize(source):
    # Convert it to grayscale
    grey = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization with the function equalizeHist
    hist = cv2.equalizeHist(grey)

    return hist


# Apply mask to extract facial features
def apply_mask(source):
    mask = cv2.imread("mask2.png")
    grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    masked_img = cv2.bitwise_and(source, grey)

    return masked_img

def create_filters():
    filters = []
    for n in range(1, 8):
        for m in range(1,5):
            kern = cv2.getGaborKernel((11, 11), 4.0, (n - 1) * np.pi / 8, 2 ** (0.5), 4 * (2 ** (.5 * (m - 1))), 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

def process(source, filters):
    accum = np.zeros_like(source)
    for kern in filters:
        fimg = cv2.filter2D(source, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def pca(source):
    """ Principal Component Analysis
    input: source, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance and mean.
    """
    mean, eigenvectors = cv2.PCACompute(source)
    print mean.shape
    print eigenvectors.shape
    cv2.imshow("source", source)
    cv2.imshow("pca", eigenvectors)

    return

if __name__ == "__main__":
    main()
