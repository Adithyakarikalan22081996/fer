import cv2
import numpy as np
from keras.models import load_model

model = load_model('C:\\Users\\91967\\PycharmProjects\\fer\\emotion_model.hdf5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

video_path = input("Enter the path to the video file: ")
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray,
                                                                                                                  scaleFactor=1.1,
                                                                                                                  minNeighbors=5,
                                                                                                                  minSize=(
                                                                                                                  30,
                                                                                                                  30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))
        face = np.reshape(face, (None, 64, 64, 1))
        face = face / 255.0  # Normalize

        emotion_probs = model.predict(face)
        predicted_emotion = emotion_labels[np.argmax(emotion_probs)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Facial Emotion Recognizer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()