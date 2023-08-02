import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime 

path='image'
image=[]
classNames= []
myList=os.listdir(path)

for cls in myList:
    cur=cv2.imread(f'{path}/{cls}')
    image.append(cur)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)    
     
def findencoding(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist    

def markattendance(name):
    with open('attendance.csv', 'r') as f:
        myDataList = f.readlines()
        namelist = []
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0].strip())
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            with open('attendance.csv', 'a') as f:
                f.writelines(f'{name},{dtString}\n')
        print(myDataList)


 
# markattendance('Elon')        


encodeListKnown = findencoding(image)
print('encoding complete')

cap=cv2.VideoCapture(0)
while True:
    success , img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # encode=face_recognition.face_encodings(img)[0]

    facescurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facescurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facescurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        # print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)


    cv2.imshow ('Webcam',img)  
    cv2.waitKey(1) 
