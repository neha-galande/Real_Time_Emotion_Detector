import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_frame, 1.3, 5)
    
    face_emotions = []  # Store predicted emotions for all faces
    
    try:
        for (x, y, w, h) in faces:
            fc = gray_frame[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            face_emotions.append(pred)  # Append predicted emotion for each face
            
        # Pad the emotions list if the number of emotions is less than the number of detected faces
        while len(face_emotions) < len(faces):
            face_emotions.append('No face detected')  # Append 'No face detected' for any additional faces
            
        return face_emotions if face_emotions else ['No face detected'] * len(faces)  # Return emotions for all faces or 'No face detected'
    except:
        return ['Unable to detect'] * len(faces)


def show_webcam():
    face_cascade = cv2.CascadeClassifier('C:/Users/Neha/Desktop/REAL_TIME_ED/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get predicted emotions for all faces in the frame
        predicted_emotions = detect_emotion(frame)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for idx, (x, y, w, h) in enumerate(faces):
            # Display rectangle around faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Display emotion text above the rectangle
            if idx < len(predicted_emotions):  # Ensure index is within range
                emotion_text = predicted_emotions[idx]
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        cv2.imshow('Video', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')
label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('C:/Users/Neha/Desktop/REAL_TIME_ED/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a.json", "model_weights.h5")

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def start_realtime_detection():
    show_webcam()

realtime_btn = Button(top, text="Start Real-Time Detection", command=start_realtime_detection, padx=10, pady=5)
realtime_btn.configure(background="#364156", foreground='white', font=('arial', 12, 'bold'))
realtime_btn.pack(side='bottom', pady=20)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()
