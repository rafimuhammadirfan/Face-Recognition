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
from torchvision import transforms


st.markdown(
    """
    <div style='text-align: center;'>
        <h1>WEBCAM FACE RECOGNIZER</h1>
        <p>Pilih camera yang akan digunakan pada sidebar dan nyalakan camera untuk memulai fitur pengenalan wajah.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Tabs
tab1, tab2 = st.tabs([":material/photo_camera_front: Face recognition", ":material/settings: setting"])

# Load your models
with tab2:
    camera_options = ["cam 1", "cam 2", "cam 3"]
    option = st.selectbox(
        "Pilih Kamera:",
        camera_options,
        index=0,
        placeholder="Pilih kamera untuk digunakan...",
    )
    camera_index = camera_options.index(option)

with tab1:
    with open('data/svm_model1.pkl', 'rb') as file:
        model = pickle.load(file)
    df = pd.read_csv('data/classes1.csv')
    labels = df['labels'].values
    label_encoder = LabelEncoder()
    label_encoder.classes_ = labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    facenet= InceptionResnetV1(pretrained='vggface2').eval().to('cuda:0')

    def get_embedding(face_img):

        face_img = cv2.resize(face_img, (160,160))
        face_img =face_img /255.0
        face_img = np.transpose(face_img, (2,0, 1))
        face_tensor = torch.tensor(face_img, dtype=torch.float32).unsqueeze(0).to('cuda:0')
        with torch.no_grad():
            embedding = facenet(face_tensor)
        return embedding.cpu().numpy().flatten()

    def recognize_faces(faces, model, label_encoder):
        recognized_faces = []
        for face, bbox in faces:
            face_resized = cv2.resize(face, (160, 160))
            face_resized = np.float32(face_resized)
            # Convert to a tensor for FaceNet
            face_embedding = get_embedding(face_resized)
            face_embedding = face_embedding.reshape(1, -1)
            yhat_class = model.predict(face_embedding)
            yhat_prob = model.predict_proba(face_embedding)
            predicted_name = label_encoder.inverse_transform(yhat_class)[0]
            confidence = np.max(yhat_prob) * 100
            recognized_faces.append((predicted_name, confidence, bbox))
        return recognized_faces

    # Camera toggle
    on = st.toggle(":material/photo_camera:")

    if on:
        with st.spinner('Tunggu sampai kamera menyala...'):
            time.sleep(5)

            cap = cv2.VideoCapture(int(f"{camera_index}"))
            yolo_model = YOLO('data/best1.engine')#.to('cuda:0')

            stframe = st.empty()
            prev_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (1020, 800))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use YOLO to detect faces
            results = yolo_model.predict(frame, conf=0.5, iou=0.5, classes=0, augment=False, stream=True, device=0)
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
                if confidence < 70:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.putText(frame, f"tidak terdeteksi ({confidence:.2f}%)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f}%)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # FPS Calculation

            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = 1 / time_diff
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            stframe.image(frame, channels="RGB")
        cap.release()