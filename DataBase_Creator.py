import numpy as np
import cv2
from pip._vendor.distlib.compat import raw_input



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

id = raw_input('Enter user id') #rw is raw input
i_n = 1 #i_n is initial number is number of users face detected initally at 1

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        i_n += 1
        cv2.imwrite("Data_Base/user."+str(id)+"."+str(i_n)+".png",gray[x:x+w,y:y+h])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
       

    cv2.imshow('img',img)
    cv2.waitKey(30) & 0xff
    if i_n>200:
        break

cap.release()
cv2.destroyAllWindows()

