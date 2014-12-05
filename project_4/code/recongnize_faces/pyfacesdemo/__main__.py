from pyfaces import pyfaces
import cv2
import time
from PIL import Image
import sys
import time
from collections import Counter


def identify_person(names):
    mcommon = [ite for ite, it in Counter(names).most_common(1)]
    return mcommon


if __name__ == "__main__":
    try:
        start = time.time()
        argsnum = len(sys.argv)
        # print "args:",argsnum
        if(argsnum < 3):
            print "more arguments"
            sys.exit(2)

        faceCascade = cv2.CascadeClassifier(
            'pyfacesdemo/haarcascade_frontalface_alt2.xml')
        video = cv2.VideoCapture(0)
        ret, frame = video.read()
        index = 0
        person_list = []

        var = raw_input("are you new? (y or n) ")
        if var == 'y':
            # must train face first
            name = raw_input("what would you like to be called by: ")
            print "Please stand back while we take your picture"
            time.sleep(5)
            # puts person into database
            while index < 10:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray,
                    1.1,
                    5,
                    0 | cv2.cv.CV_HAAR_SCALE_IMAGE,
                    (30, 30)
                )

                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    imgname = gray[y: y + 100, x: x + 100]
                    dirname = sys.argv[1]
                    egfaces = int(sys.argv[2])
                    thrshld = float(sys.argv[3])
                    im = Image.fromarray(imgname)
                    im.save('faces_db2/%s_%d.png' % (name, index))
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the resulting frame
                cv2.imshow('Video', frame)
                cv2.waitKey(1)

                # Capture frame-by-frame
                ret, frame = video.read()
                index = index + 1

            print "You are now in our database"

        else:
            # trys to recognize the person in 15 frames
            while index < 15:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray,
                    1.1,
                    5,
                    0 | cv2.cv.CV_HAAR_SCALE_IMAGE,
                    (30, 30)
                )

                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    imgname = gray[y: y + 100, x: x + 100]
                    dirname = sys.argv[1]
                    egfaces = int(sys.argv[2])
                    thrshld = float(sys.argv[3])
                    im = Image.fromarray(imgname)

                    im.save('face.png')
                    pyf = pyfaces.PyFaces(
                        'face.png', dirname, egfaces, thrshld)
                    # parses string to find name from the image
                    string = pyf.match
                    name1 = string.split("/")
                    name2 = name1[-1].split("_")
                    person_list.append(name2[0])
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the resulting frame
                cv2.imshow('Video', frame)
                cv2.waitKey(1)

                # Capture frame-by-frame
                ret, frame = video.read()
                index = index + 1

            # When everything is done, release the capture
            video.release()
            cv2.destroyAllWindows()

            person = identify_person(person_list)
            print "You are " + person[0]
            end = time.time()
            print 'took :', (end - start), 'secs'

    except Exception, detail:
        print detail.args
        print "usage:python pyfacesdemo dirname numofeigenfaces threshold"
