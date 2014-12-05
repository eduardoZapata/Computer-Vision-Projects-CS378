import cv2
import numpy as np


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
show_hsv = False
show_glitch = False
cur_glitch = prvs.copy()

while(1):
    ret, frame2 = cap.read()

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, gray, 0.5, 1, 3, 15, 3, 5, 1)
    prevgray = gray

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    # locate face
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # selected region of interest
        face_gray = gray[y:y + h, x:x + w]
        # mouth = mouth_cascade.detectMultiScale(face_gray)
        # for (mx, my, mw, mh) in mouth:
        #     face_mouth = face_gray[my:my+mh, mx:x+w]
    # create flow diagram
    cv2.imshow('flow', draw_flow(face_gray, flow))
    if show_hsv:
        cv2.imshow('flow HSV', draw_hsv(flow))
    if show_glitch:
        cur_glitch = warp_flow(cur_glitch, flow)
        cv2.imshow('glitch', cur_glitch)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
    if ch == ord('1'):
        show_hsv = not show_hsv
        # shows hsv
        print 'HSV flow visualization is', ['off', 'on'][show_hsv]
    if ch == ord('2'):
        show_glitch = not show_glitch
        if show_glitch:
            cur_glitch = frame2.copy()
        print 'glitch is', ['off', 'on'][show_glitch]
    if ch == ord('q'):
        # press q to quite
        break

cap.release()
cv2.destroyAllWindows()
