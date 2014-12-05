import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

cap = cv2.VideoCapture(0)
ret, img = cap.read()


found = False
#boundaries
#
#	o - - - - o
#	|		  |
#	|		  |
#	o - - - - o
#
l_x = 0
l_y = 0
r_x = 0
r_y = 0
l_xh = 0
l_yh = 0
r_xh = 0
r_yh = 0

while ret:
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(gray_img, 1.05, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		face_gray = gray_img[y:y+h, x:x+w] 
		face_color = img[y:y+h, x:x+w] 
		eyes = eye_cascade.detectMultiScale(face_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(face_color, (ex,ey), (ex+ew, ey+eh), (255,255,0), 2)
			# track nose 
			nose = nose_cascade.detectMultiScale(face_gray)
			for (nx,ny,nw,nh) in nose:
				if ny > ey + eh:
					cv2.rectangle(face_color,(nx,ny),(nx+nw,ny+nh), (0,0,0),2)
					# track mouth
					mouth = mouth_cascade.detectMultiScale(face_gray)
					for point in mouth:
						xx, yy, ww, hh = point
						if yy > ny + nh:
							print point
							face_mouth = face_color[yy:yy+hh, xx:xx+w]
							cv2.rectangle(face_color, (xx,yy),(x+w,yy+hh), (255,0,255),2)
	
	cv2.imshow("frame", img)
	cv2.waitKey(0)
	ret, img = cap.read()

cv2.destroyAllWindows()
cap.release

# corners = cv2.goodFeaturesToTrack(face_mouth, 10, 0.01,10)
# corners = np.int0(corners)

# for i in corners:
# 	x,y = i.ravel()
# 	cv2.circle(img, (x,y), 3, 255,-1)

