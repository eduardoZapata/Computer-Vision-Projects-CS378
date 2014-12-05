Project 4
=========

1. Facial Recognition
	Located in recongnize_faces:
	In pyfacesdemo we have our main file called __main__.py, this is where most of our 	code for running facial recogntion is. We call pyfaces within that file which runs 	the Eigenfaces algorithm with the given database
	HOW TO RUN FACIAL RECOGNITION PROGRAM: 
	cd into recongnize_faces
	to compile and run type command:  python pyfacesdemo <pwd>/faces_db2 4 3
	the parameters are the path to the data base, and the threshold parameters
	This currently uses a webcam, but can be changed to a video

2. Eye Tracking

	Located in eye_tracking:
 
        TO RUN:
	cd into eye_tracking
	to compile and run type command:  python eyes.py

	the program will then run the algorithm using the eyes.mp4 video located in the eye_tracking/test_data/ directory
        All support files are located within eye_tracking, and eye_tracking/test_data

	We take the video and for each frame we apply eyes cascade in order to establish a region of interest around the eyes and then run the Hough circle algorithm on that. From there, we determine which circle corresponds to the right eye and then we add that to a list. That list is then used to compare the current frame to a tunable previous frame. This way we can detect changes in the x and y coordinate values and determine change in direction. 


3. Expression Detection
   Located in recognize_emotion: 
   Here is where our several attempts of recognizing expression is found.
   The file op_flow.py will track a face and compute the optical flow within that given region. 
   TO RUN:
   python op_flow.py , this uses a webcam to track the face.
   press 1 to show hsv
   press q to quit

   Other attempts:
   Located in recongnize_faces/pyfacesdemo
   emotion.py:
   Emotion.py creates a set of eigenfaces based of a curated database of faces with "happy" or "neutral" expressions. The photos are cropped in half in order to disregard any extraneous features. The eigenfaces are then placed in recongnize_faces/eigenfaces. The best eigenfaces is then put into emotion_db and then the program can be run with the following.

   If the implementation was more reliable, a unit test could be created that tests the matching against a photo of a person and it would be required to match it correctly.
   

   TO RUN:
   Go to recongnize_faces/pyfacesdemo/
   python pyfacesdemo/ /absolute_path_to/recongnize_faces/emotion_db/ 4 3
   Click no
   Smile at the camera!

   This implementation is very inconsistent in that once it matches a person to a face in the database, it is very reluctant to match it to another, even with a change of facial gesture.

   emotion_attempt_1.py:
   This implementation attempts to the follow the paper linked [here](http://cs229.stanford.edu/proj2009/AgrawalCosgriffMudur.pdf). It applies some preprocessing to the static images which creates a set of masked images. A gabor filter are then applied to these masked images. The roadblock here was applying PCA to the images, which produced the final images in the paper. 






