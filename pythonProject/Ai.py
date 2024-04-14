import os
import pickle
import cv2
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL" : "https://facedectectionh-default-rtdb.firebaseio.com/",
    "storageBucket" : "gs://facedectectionh.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

print("Loading Encode File ...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIDs = pickle.load(file)
encodeListKnown, studentIDs = encodeListKnownWithIDs
print("Encode File Loaded")

modeType = 0
counter = 0
ids = []
imgStudents = []

# Filter out None values from 'encodeListKnown'
encodeListKnown = [encode for encode in encodeListKnown if encode is not None]

# Now we can safely check for the reference shape if 'encodeListKnown' is not empty
reference_shape = encodeListKnown[0].shape if encodeListKnown else None

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
new_width = int(height * (16/10))
new_height = int(height)

# Đặt kích thước cửa sổ
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", new_width, new_height)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurName = face_recognition.face_locations(imgS)
    encodeCurName = face_recognition.face_encodings(imgS, faceCurName)

    ids = []
    imgStudents = []

    for encodeFace, faceLoc in zip(encodeCurName, faceCurName):
        # Convert encodeFace to a numpy array if it's not already one
        encodeFace = np.array(encodeFace) if not isinstance(encodeFace, np.ndarray) else encodeFace
        # Ensure encodeFace has the same shape as the reference shape
        if reference_shape and encodeFace.shape == reference_shape:
            # Now we can safely perform the subtraction operation
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = (x1, y1, x2 - x1, y2 - y1)

                ids.append(studentIDs[matchIndex])
                studentInfo = db.reference(f'Students/{studentIDs[matchIndex]}').get()
                imgStudents.append(studentInfo)

                # Draw rectangle around the face
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

                # Display student information
                # Trong vòng lặp while, phần hiển thị thông tin sinh viên
                cv2.putText(img, f"Name: {studentInfo['Name']}", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, f"ID: {studentInfo['ID']}", (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, f"Major: {studentInfo['Major']}", (bbox[0], bbox[1] + bbox[3] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, f"Intake: {studentInfo['Intake']}", (bbox[0], bbox[1] + bbox[3] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    for id, imgStudent in zip(ids, imgStudents):
        print(f"ID: {id}, Student Info: {imgStudent}")

    cv2.imshow("Face Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()