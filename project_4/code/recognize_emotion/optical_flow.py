import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read() 
# faces = face_cascade.detectMultiScale(gray_img, 1.05, 5)
# for (x,y,w,h) in faces:
# 		cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
# 		face_prev = gray_img[y:y+h, x:x+w] 
# 		face_color = frame1[y:y+h, x:x+w] 

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while True:

	ret, img = cap.read()

	next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	# faces = face_cascade.detectMultiScale(gray_img, 1.05, 5)

	# for (x,y,w,h) in faces:
	# 	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	# 	face_next = gray_img[y:y+h, x:x+w] 
	# 	face_color = img[y:y+h, x:x+w] 

	flow = cv2.calcOpticalFlowFarneback(prvs,next,0.5,1,3,15,3,5,1)
	print flow
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	cv2.imshow('frame2',rgb)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	elif k == ord('s'):
		cv2.imwrite('opticalfb.png', img)
		cv2.imwrite('opticalhsv.png',rgb)
	prvs = next

	# cv2.imshow("cats", img)
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break


cap.release()
cv2.destroyAllWindows()