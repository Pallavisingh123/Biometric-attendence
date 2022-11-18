import cv2
import numpy as np
import face_recognition

imgSona = face_recognition.load_image_file('images/Pallavi.jpg')
imgSona= cv2.cvtColor(imgSona,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images/Shivani.jpeg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSona)[0]
encodeSona = face_recognition.face_encodings(imgSona)[0]
# cv2.rectangle(imgSona,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]) (255,0,255),2)
cv2.rectangle(imgSona, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgSona)[0]
encodeTest = face_recognition.face_encodings(imgSona)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeSona],encodeTest)
faceDis = face_recognition.face_distance([encodeSona],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow("sonali original ",imgSona)
cv2.imshow("sonali test",imgTest)
cv2.waitKey(0)