import cv2
import sys
from PIL import Image

def scale(source, face): # eye1, eye2, mouth
    crop_img = source[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
    resized_image = cv2.resize(crop_img, (200, 200))

    cv2.imshow("a", resized_image)
    cv2.waitKey(10)

    return resized_image

faceCascade = cv2.CascadeClassifier('pyfacesdemo/haarcascade_frontalface_alt2.xml')

video_capture = cv2.VideoCapture(0)
i = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        scaled_img = scale(frame, (x, y, w, h))
        # im = Image.fromarray(scaled_img)
        cv2.imwrite('emotion/happy_%d.png' % i, scaled_img)
        i += 1
    



     # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
