import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import pickle
from sklearn.preprocessing import LabelEncoder
from facenet_pytorch import InceptionResnetV1
import pandas as pd
import torch
import threading
from datetime import datetime

st.markdown(
    """
    <div style='text-align: center;'>
        <h1>CCTV FACE RECOGNIZER</h1>
        <p>Pilih kamera yang akan digunakan pada menu setting dan nyalakan kamera untuk memulai fitur pengenalan wajah.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Tabs
tab1, tab2 = st.tabs([":material/photo_camera_front: Face recognition", ":material/settings: setting"])

# Tab untuk input model dan kamera
with tab2:
    model_fr = st.text_input(
        "Masukkan SVM model:", 
        "data/svm_model.pkl", 
        help="Masukkan jalur file model SVM yang telah dilatih sebelumnya. Model ini digunakan untuk melakukan prediksi atau klasifikasi berdasarkan dataset yang dilatih. Format file biasanya berupa .pkl."
    )
    
    model_label = st.text_input(
        "Masukkan label model:", 
        "data/classes.csv", 
        help="Masukkan jalur file yang berisi label atau kelas yang digunakan dalam model SVM. File ini biasanya berupa file CSV yang berisi nama-nama kelas yang akan diprediksi oleh model."
    )
    
    cam = st.text_input(
        "Masukkan URL Kamera:", 
        help="Masukkan URL atau alamat untuk sumber kamera yang akan digunakan untuk menangkap video. Jika menggunakan kamera eksternal atau IP, masukkan URL yang sesuai, seperti rtsp://username:password@ip_address:port."
    )
    cnf = st.slider("Adjust Confidence", 0, 100, 85)
with tab1:
    with open(model_fr, 'rb') as file:
        model = pickle.load(file)
    df = pd.read_csv(model_label)
    labels = df['labels'].values
    label_encoder = LabelEncoder()
    label_encoder.classes_ = labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Helper Functions
    def get_embedding(face_img):
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img / 255.0
        face_img = np.transpose(face_img, (2, 0, 1))
        face_tensor = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = facenet(face_tensor)
        return embedding.cpu().numpy().flatten()

    def recognize_faces(faces, model, label_encoder):
        recognized_faces = []
        for face, bbox in faces:
            face_embedding = get_embedding(face)
            face_embedding = face_embedding.reshape(1, -1)
            yhat_class = model.predict(face_embedding)
            yhat_prob = model.predict_proba(face_embedding)
            predicted_name = label_encoder.inverse_transform(yhat_class)[0]
            confidence = np.max(yhat_prob) * 100
            recognized_faces.append((predicted_name, confidence, bbox))
        return recognized_faces

    # Detection History
    detection_history = []

    # Camera Toggle
    on = st.toggle("camera")
    
    if on:
        with st.spinner('Tunggu sampai kamera menyala...'):
            time.sleep(5)

            cap = cv2.VideoCapture(cam, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            yolo_model = YOLO('data/best1.engine')

            stframe = st.empty()
            empty = st.empty()
            prev_time = 0
        
            while True:
                if not cap.isOpened():
                    empty.warning("Kamera terputus, mencoba menghubungkan kembali...")
                    time.sleep(5)  # Tunggu sebelum mencoba lagi
                    cap = cv2.VideoCapture(cam, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
                    continue

                ret, frame = cap.read()
                if not ret:
                    empty.warning("Gagal membaca frame, mencoba menghubungkan kembali...")
                    time.sleep(5)  # Tunggu sebelum mencoba lagi
                    cap.release()
                    cap = cv2.VideoCapture(cam, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
                    continue
                frame = cv2.resize(frame, (1000,800))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Use YOLO to detect faces
                results = yolo_model.predict(frame, conf=0.5, iou=0.5, classes=0, augment=False, device=0)
                faces = []
                for result in results:
                    for box in result.boxes:
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                        face = frame[y_min:y_max, x_min:x_max]
                        if face.size != 0:
                            faces.append((face, (x_min, y_min, x_max, y_max)))

                # Recognize the detected faces
                recognized_faces = recognize_faces(faces, model, label_encoder)

                for name, confidence, bbox in recognized_faces:
                    x_min, y_min, x_max, y_max = bbox
                    if confidence >= cnf:
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({confidence:.2f}%)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.putText(frame, f"tidak terdeteksi ({confidence:.2f}%)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Add to detection history
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        detection_history.append({"Name": name, "Last Detected": now})

                curr_time = cv2.getTickCount()
                time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
                fps = 1 / time_diff
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                stframe.image(frame, channels="RGB")

                # Display detection history in a table
        cap.release()

        df_detection = pd.DataFrame(detection_history)
        st.dataframe(df_detection)

