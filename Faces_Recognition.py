# Since haar_cascade has been used the predicted output might not be the right one

import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier(r'C:\Users\ghode\Python\OpenCV\haar_face.xml')

# features = np.load('C:\Users\ghode\Python\features.npy')
# labels = np.load('C:\Users\ghode\Python\labels.npy')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\ghode\Python\face_trained.yml')

img =cv.imread(r'C:\Users\ghode\Python\val\ben_afflek\b.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)  

for(x,y,w,h) in faces_rect:
    faces_rei = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_rei)
    print(f'Label= {people[label]} with a confidence of {confidence}')
    
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0),2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)

cv.imshow('Detected Face', img)

cv.waitKey(0)