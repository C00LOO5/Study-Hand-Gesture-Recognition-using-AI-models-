import cv2
import mediapipe as mp
import numpy as np
import joblib
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

svm = joblib.load("svm_model.pkl")
knn = joblib.load("knn_model.pkl")
rf = joblib.load("rf_model.pkl")
cnn = load_model("cnn_model.keras")
lstm = load_model("lstm_model.keras")
transformer = load_model("transformer_model.keras")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            if len(landmarks) == 42:  
                X_ml = np.array(landmarks).reshape(1, -1)
                X_ml_scaled = scaler.transform(X_ml)
                
                pred_svm = label_encoder.inverse_transform(svm.predict(X_ml))[0]
                pred_knn = label_encoder.inverse_transform(knn.predict(X_ml))[0]
                pred_rf = label_encoder.inverse_transform(rf.predict(X_ml))[0]
                
                X_dl = np.array(landmarks).reshape(1, 21, 2)
                
                pred_cnn = label_encoder.inverse_transform([np.argmax(cnn.predict(X_dl))])[0]
                pred_lstm = label_encoder.inverse_transform([np.argmax(lstm.predict(X_dl))])[0]
                pred_transformer = label_encoder.inverse_transform([np.argmax(transformer.predict(X_dl))])[0]
                
                cv2.putText(frame, f"SVM: {pred_svm}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"KNN: {pred_knn}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"RF: {pred_rf}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"CNN: {pred_cnn}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"LSTM: {pred_lstm}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Transformer: {pred_transformer}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
