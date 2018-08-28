import numpy as np
import cv2

detector= cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#cap = cv2.imread('image.jpg')
rec=cv2.createLBPHFaceRecognizer()
rec.load("recognizer/trainningData.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,3,1,0,4)
while(True):
    img = cv2.imread('image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.4,3)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
        id,conf=rec.predict(gray[y:y+h,x:x+w]) 
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);
        
    #cv2.namedWindow('frame',cv2.cv.CV_WINDOW_AUTOSIZE)
    small = cv2.resize(img, (0,0), fx=0.9, fy=0.9)
    cv2.imshow('output',small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
    
cap.release()
cv2.destroyAllWindows()
