from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
#from keras.models import load_model


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

detect_model_path = 'data/haarcascade_frontalface_default.xml'


face_detection = cv2.CascadeClassifier(detect_model_path)
emotion_classifier = load_model('data/recog.h5')
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

cv2.namedWindow('cam')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #frame = imutils.resize(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype = "uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse = True, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        img = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(img, (48, 48))
        roi = roi.astype("float")/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)

        pred = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(pred)
        label = EMOTIONS[pred.argmax()]
        print(label)

        #font = cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8  # Creates a font
        x = 100  # position of text
        y = 200  # position of text
        cv2.putText(frameClone, label, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)



    cv2.imshow('cam', frameClone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
