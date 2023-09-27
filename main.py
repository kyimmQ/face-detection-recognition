import face_recognition
import os
import cv2
import numpy as np
import telegram
import asyncio
import datetime

async def alert(img, name):
    my_token = ""
    bot = telegram.Bot(token=my_token)
    now = datetime.datetime.now().strftime('%H:%M:%S')
    async with bot:
        await bot.sendPhoto(chat_id="", photo = img, caption=f'{name} {now}')
        
path = "image"
images = []
names = []
myList = os.listdir(path)

for img in myList:
  curimg = cv2.imread(f'{path}/{img}')
  images.append(curimg)
  names.append(os.path.splitext(img)[0])
  
def findEncodings(images):
  encodeList = []
  for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeList.append(encode)
  return encodeList
encodeListKnown = findEncodings(images)

cap = cv2.VideoCapture(0)
attendance_list = []
while True:
    success, img = cap.read()
    imgS= cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesInCurFrame = face_recognition.face_locations(imgS)
    encodeOfCurFrame = face_recognition.face_encodings(imgS,facesInCurFrame)
    
    for encodedFace, loc in zip(encodeOfCurFrame, facesInCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodedFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodedFace)
        match_index = np.argmin(faceDis)
        if matches[match_index]:
            name = names[match_index].upper()
            y1,x2,y2,x1 = loc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            if name not in attendance_list:
                cv2.imwrite("alert.png", img)
                asyncio.run(alert("alert.png", name))
                attendance_list.append(name)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()