import cv2
import numpy as np
import face_recognition

imgE1 = face_recognition.load_image_file('image/elon.jpg')
imgE1 = cv2.cvtColor(imgE1, cv2.COLOR_RGB2BGR)
faceLoc1=face_recognition.face_locations(imgE1)[0]
encE1=face_recognition.face_encodings(imgE1)[0]
cv2.rectangle(imgE1,(faceLoc1[0],faceLoc1[3]),(faceLoc1[1],faceLoc1[2]),(255,0,255),2)



imgE2 = face_recognition.load_image_file('image/bill.jpg')
imgE2 = cv2.cvtColor(imgE2, cv2.COLOR_RGB2BGR)
faceLoc2=face_recognition.face_locations(imgE2)[0]
encE2=face_recognition.face_encodings(imgE2)[0]
cv2.rectangle(imgE2,(faceLoc2[0],faceLoc2[3]),(faceLoc2[1],faceLoc2[2]),(255,0,255),1)
 
result=face_recognition.compare_faces([encE1],encE2)
faceD=face_recognition.face_distance([encE1],encE2)
print(result,faceD)
cv2.putText(imgE1,f'{result} {round(faceD[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk', imgE1)
cv2.imshow('Other elon',imgE2)
cv2.waitKey(0)
