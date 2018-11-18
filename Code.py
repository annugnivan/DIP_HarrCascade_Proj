#facial tracking using python and opencv
import numpy as np
import cv2
#loads in the Haar's cascade for the face and eye
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyedetect = cv2.CascadeClassifier('haarcascade_eye.xml')
#import utilities for making video
record = cv2.VideoWriter_fourcc(*'XVID')
#video will be recorded in 20 fps with quality of 640*480px
vidrec = cv2.VideoWriter('recorded.avi',record,20.0, (640,480))
#the video capture object
vidcap = cv2.VideoCapture(0)
#load in the image
img = cv2.imread('manface.jpg')
#convert image from color to greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#to detect the face for image
img4face = facedetect.detectMultiScale(gray, 1.3, 5)
#places rectangle of blue around face
for (x,y,w,h) in img4face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#makes blue rectangle
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    img4eye = eyedetect.detectMultiScale(roi_gray)
    for (xx,yy,ww,hh) in img4eye:
        cv2.rectangle(roi_color,(xx,yy),(xx+ww,yy+hh),(0,255,0),2)#makes green rectangle
while (True):
    #records each individual frame
    ret, frame = vidcap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #integer arguments set the distance for the facial recognition
    face = facedetect.detectMultiScale(gray, 1.3, 5) #optimal parameters for tracking
    #Haar's cascade for the face detection and eye
    #specificially for face
    for (x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#blue rect in color if got face
            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)#blue rect in color if got face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            #tracks the presence of eyes on the face
            #placed inside for loop so detects eye inside of face
            eye = eyedetect.detectMultiScale(roi_gray) 
            for (xx, yy, ww, hh) in eye:
                cv2.rectangle(roi_color,(xx,yy),(xx+ww,yy+hh),(0,255,0),2)#green rectif got eyes
    #records the coloured frame
    vidrec.write(frame)
    #shows the recording in a new window
    cv2.imshow('Colour',frame)#pqrints frame in colour
    cv2.imshow('Grayscale',gray)#prints frame in 
    cv2.imshow('img',img)
    #breaks the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#since loop is broken, webcam shuts off and closes recording
vidcap.release()
vidrec.release()
cv2.destroyAllWindows()
