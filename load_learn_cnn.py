# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 21:22:51 2018
https://medium.com/@jinilcs/a-simple-keras-model-on-my-laptop-webcam-dda77521e6a0
@author: admin
"""

# part 3 : making new predictions
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2
from PIL import Image

"""
# returns a compiled model
# identical to the previous one
start = time.time()
model = load_model('learn_cnn.h5')
test_image = image.load_img('dataset/single_prediction/Abdullah_Gul_8.jpg', target_size=(128, 128))
test_image = image.img_to_array(test_image)
# we have four dimentions (1,128, 128,3) 
#Our keras model used a 4D tensor, (images x height x width x channel)
#So changing dimension 128x128x3 into 1x128x128x3 
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
# to know number 1 is dogs or cats, 0 is dogs or cats
if result[0][0] == 1:
    prediction = 'yes'
else:
    prediction = 'no'
#how to improve execution time
end = time.time()
print("CNN : ", format(end - start, '.2f'))
"""

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.

def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        cv2.putText(frame, "I am Hung", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
    return frame # We return the image with the detector rectangles.


#Load the saved model
model = load_model('learn_cnn.h5')
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()
        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((128,128))
        img_array = np.array(im)
        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)
        #Calling the predict method on model to predict 'me' on the image
        prediction = int(model.predict(img_array)[0][0])
        canvas = frame
        # to know number 1 is dogs or cats, 0 is dogs or cats
        if prediction == 1:
            prediction = 'yes'
           
        else:
            prediction = 'no'
            cv2.putText(frame, "Văn Hùng", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
            canvas = detect(gray, frame) # We get the output of our detect function.
            
        cv2.imshow("Capturing", canvas)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()