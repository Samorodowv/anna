import face_recognition
import numpy as np
from imutils.video import VideoStream
import cv2
import pickle

def meet():
    answer = input("Lets meet? (Y/n)\n")
    return answer.lower() == "y"

print("Loading known face image(s)")
obama_image = face_recognition.load_image_file("obama_small.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

try:
    with open("trained_knn_model.clf", 'rb') as f:
        knn_clf = pickle.load(f)
except FileNotFoundError:
    print("Need to train trained_knn_model.clf")

face_locations = []
face_encodings = []
vs = VideoStream(0).start()
while True:
    try:
        output = cv2.resize(vs.read(), (320, 240))
        face_locations = face_recognition.face_locations(output)
        if len(face_locations) < 1:
            continue
        face_encodings = face_recognition.face_encodings(output, face_locations)
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
            name = "unknown"
            if match[0]:
                name = "Barack Obama"
            if name == "unknown":
                meet()
            print("I see someone named {}!".format(name))
    except KeyboardInterrupt:
        exit()
