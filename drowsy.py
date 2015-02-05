#!/usr/bin/python
import cv2
import time
import Image
import os
import numpy as np
import time
import pygame

#----------------------------------------------------------------------------
# Face Detection Test based on OpenCV)
# Modified/using examples from:
# 
# http://japskua.wordpress.com/2010/08/04/detecting-eyes-with-python-opencv
#----------------------------------------------------------------------------

#not needed 
#pygame.init()
#pygame.mixer.init()

FREQ = 44100   # same as audio CD
BITSIZE = -16  # unsigned 16 bit
CHANNELS = 2   # 1 == mono, 2 == stereo
BUFFER = 1024  # audio buffer size in no. of samples
FRAMERATE = 1000 # how often to check if playback has finished
pygame.mixer.init(FREQ, BITSIZE, CHANNELS, BUFFER)
soundfile = 'error2.mp3'
clock = pygame.time.Clock()
pygame.mixer.music.load(soundfile)


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
l_eye = cv2.CascadeClassifier('haarcascades/haarcascade_lefteye_2splits.xml')
r_eye = cv2.CascadeClassifier('haarcascades/haarcascade_righteye_2splits.xml')

capture = cv2.VideoCapture(0)
capture.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, 160)
capture.set(cv2.cv.CV_CAP_PROP_EXPOSURE, 20.0)
name = 'detect'

cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
ocur=0

while True:
	
	s, img = capture.read()
	if s <> None:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 3) #hmm, 5 required neighbours is actually a lot.
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1) # if you want colors, don't paint into a grayscale...
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
			eyes = r_eye.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
			#print len(eyes)
			if len(eyes) == 0:
				ocur+=1
				if ocur==1:
					t1=time.time()			
			if ocur > 1:
				t2=time.time()
				ocur=0
				if t2-t1 < 2:
					pygame.mixer.music.play()
					while pygame.mixer.music.get_busy():
						clock.tick(FRAMERATE)
		cv2.imshow(name, img)    

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()

