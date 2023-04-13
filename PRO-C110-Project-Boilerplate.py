# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf


mymodel=tf.keras.models.load_model("keras_model.h5")

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		myimage=cv2.resize(frame(224,224))

		test_image=np.array(myimage,dtype=np.float32)
		
		# expand the dimensions
		expand_image=np.expand_dims(test_image,axis=0)
		
		# normalize it before feeding to the model
		normalised_image=expand_image/255.0
		
		# get predictions from the model
		predication=mymodel.predict(normalised_image)
		print("predication:  ",predication)
		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
