import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

cap = cv2.VideoCapture(0)
ret, img = cap.read()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
faces = face_cascade.detectMultiScale(gray_img, 1.05, 5)

# create a boundary around mouth
good = []
point = []
foundMouth = False
while ret:
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		face_gray = gray_img[y:y+h, x:x+w] 
		face_color = img[y:y+h, x:x+w] 
		# track nose 
		nose = nose_cascade.detectMultiScale(face_gray)
		for (nx,ny,nw,nh) in nose:
			cv2.rectangle(face_color,(nx,ny),(nx+nw,ny+nh), (0,0,0),2)
			# track mouth
			mouth = mouth_cascade.detectMultiScale(face_gray)
			print "mouth"
			print mouth
			for point in mouth:
				xx, yy, ww, hh = point
				if yy > ny + nh:
					print point
					good.append(point)
					foundMouth = True
					for (mx,my,mw,mh) in good:
						cv2.circle(face_color,(mx,my),10, (0,0,200),2)
						cv2.circle(face_color,(mx+mw,my),10, (0,0,200),2)
						cv2.circle(face_color,(mx+mw,my+mh),10, (0,0,200),2)
						cv2.circle(face_color,(mx,my+mh),10, (0,0,200),2)
				# else:
				# 	ret, img = cap.read()
			

	print good
	cv2.imshow('frame',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	ret, img = cap.read()

# ret, img = cap.read()
# while ret:

# 	for (x,y,w,h) in faces:
# 		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# 		face_gray = gray_img[y:y+h, x:x+w] 
# 		face_color = img[y:y+h, x:x+w] 
# 		# track nose 
# 		nose = nose_cascade.detectMultiScale(face_gray)
# 		for (nx,ny,nw,nh) in nose:
# 			cv2.rectangle(face_color,(nx,ny),(nx+nw,ny+nh), (0,0,0),2)
# 			# track mouth
# 			mouth = mouth_cascade.detectMultiScale(face_gray)
# 			for point in mouth:
# 				xx, yy, ww, hh = point
# 				if yy > ny + nh:
# 					print point
# 					good.append(point)
# 					cv2.circle(face_color, (xx,yy), 5, (0,255,0), 1)

	 
# 	cv2.imshow('frame',img)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()
# 	ret, img = cap.read()
	
	# check previous point

	
	# calculate optical flo
	# p1, st, err = cv2.calcOpticalFlowPyrLK(img, gray_img, p0, None, **lk_params)

    # Select good points
	# good_new = p1[st==1]
	# good_old = p0[st==1]

    # draw the tracks
	# for i,(new,old) in enumerate(zip(good_new,good_old)):
	# 	a,b = new.ravel()
	# 	c,d = old.ravel()
	# 	mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
	# frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
	# img = cv2.add(frame,mask)


    # Now update the previous frame and previous points
	# old_gray = frame_gray.copy()
	# good = good_new.reshape(-1,1,2)

cap.release()
cv2.destroyAllWindows()