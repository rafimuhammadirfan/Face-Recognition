import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import torch
from tensorflow.keras.models import load_model



# Setup tampilan Streamlit
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>CCTV PEOPLE TRACKER & AGE AND GENDER</h1>
        <p>Pilih camera yang akan digunakan pada sidebar dan nyalakan camera untuk memulai fitur PEOPLE TRACKER.</p>
    </div>
    """,
    unsafe_allow_html=True
)
tab1, tab2 = st.tabs(["Deployment", "Setting"])
with tab2:
    camera_options = ["cam 1", "cam 2", "cam 3"]
    option = st.selectbox(
        "Pilih Kamera:",
        camera_options,
        index=0,
        placeholder="Pilih kamera untuk digunakan...",
    )
    camera_index = camera_options.index(option)

    cam = st.text_input("Masukan URL camera")

with tab1:
# Load YOLO model
    model = YOLO('data/yolov8l.engine')

    # Load gender and age prediction model
    gender_dict = {1: "Female", 0: "Male"}
    age_gender_model = load_model('data/modelage&gender.h5')

    # Dictionary for storing person data
    people_data = {}

    def group_age(age):
        if age <= 15:
            return "Anak-anak (0-15 tahun)"
        elif age <= 28:
            return "Dewasa Muda (18-28 tahun)"
        elif age <= 50:
            return "Dewasa (26-40 tahun)"
        elif age <= 75:
            return "Lansia Awal (61-75 tahun)"
        else:
            return "Lansia (91-101 tahun)"

    def predict_gender_age(face):
        try:
            face_resized = cv2.resize(face, (128, 128))
            face_normalized = face_resized / 255.0
            face_input = face_normalized.reshape(1, 128, 128, 1)
            pred = age_gender_model.predict(face_input, verbose=0)
            pred_gender = gender_dict[round(pred[0][0][0])]
            pred_age = round(pred[1][0][0])
            age_group = group_age(pred_age)
            return pred_gender, pred_age, age_group
        except Exception as e:
            print("Error during prediction:", e)
            return None, None, None

    def calculate_iou(box1, box2):
        # Mengkonversi input ke tensor
        box1 = torch.tensor(box1, dtype=torch.float32)
        box2 = torch.tensor(box2, dtype=torch.float32)
        
        # Mendapatkan koordinat kiri atas dan kanan bawah dari kedua kotak
        x1, y1, x2, y2 = box1
        x1p, y1p, x2p, y2p = box2
        
        # Menghitung koordinat perpotongan (intersection)
        xi1 = torch.max(x1, x1p)
        yi1 = torch.max(y1, y1p)
        xi2 = torch.min(x2, x2p)
        yi2 = torch.min(y2, y2p)
        
        # Menghitung area intersection (tumpang tindih)
        inter_area = torch.max(torch.tensor(0.0), xi2 - xi1) * torch.max(torch.tensor(0.0), yi2 - yi1)
        
        # Menghitung area dari masing-masing kotak
        box1_area = torch.max(torch.tensor(0.0), x2 - x1) * torch.max(torch.tensor(0.0), y2 - y1)
        box2_area = torch.max(torch.tensor(0.0), x2p - x1p) * torch.max(torch.tensor(0.0), y2p - y1p)
        
        # Menghitung area union
        union_area = box1_area + box2_area - inter_area
        
        # Menghitung IoU
        iou = inter_area / union_area if union_area > 0 else torch.tensor(0.0)
        
        return iou

    def process_frame(frame):
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, conf=0.5, iou=0.4)
        a = results[0].boxes.data.cpu().numpy()
        px = pd.DataFrame(a).astype("float")
        person_count = 0

        for index, row in px.iterrows():
            x1, y1, x2, y2, score, class_id = map(int, row[:6])
            if class_id == 0:
                person_count += 1
                face = frame[y1:y2, x1:x2]
                if face.size == 0 or x1 < 0 or y1 < 0:
                    continue

                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                pred_gender, pred_age, age_group = predict_gender_age(face_gray)
                if pred_gender is None:
                    continue

                label = f"{pred_gender}, {pred_age} thn, {age_group}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                current_id = None
                for person_id, data in people_data.items():
                    iou = calculate_iou((x1, y1, x2, y2), data['bbox'])
                    if iou > 0.5:
                        current_id = person_id
                        data.update(bbox=(x1, y1, x2, y2), last_seen=time.time())
                        break
                if current_id is None:
                    current_id = len(people_data) + 1
                    people_data[current_id] = {
                        'bbox': (x1, y1, x2, y2),
                        'start_time': time.time(),
                        'last_seen': time.time(),
                        'gender': pred_gender,
                        'age': pred_age,
                        'age_group': age_group,
                        'duration': 0.0
                    }
                duration = time.time() - people_data[current_id]['start_time']
                # people_data[current_id]['gender'] = pred_gender
                # people_data[current_id]['age'] = pred_age
                # people_data[current_id]['age_group'] = age_group
                people_data[current_id]['duration'] = duration
                cv2.putText(frame, f'ID {current_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f'Persons detected: {person_count}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame, person_count

    stframe = st.empty()
    if cam:
        cap = cv2.VideoCapture(cam)
    else:
        cap = cv2.VideoCapture(int(f"{camera_index}"))  
    #cap = cv2.VideoCapture('cctv1.mp4')
    on = st.toggle("Camera", key="webcam")

    if on:
        st.subheader("Deteksi Usia, Gender, dan Durasi Per ID")
        data_display = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, person_count = process_frame(frame)

            # Filter out IDs with duration < 1 second
            df = pd.DataFrame.from_dict(
                people_data,
                orient='index',
                columns=['gender', 'age_group', 'duration',]
            )

            df['duration'] = df['duration'].apply(lambda x: float(x))  # Pastikan duration adalah float
            df = df[df['duration'] >= 2.0]  # Filter hanya yang durasi >= 1 detik

        # Konversi durasi ke format teks untuk ditampilkan
        #    df['duration'] = df['duration'].apply(lambda x: f"{x:.1f} s")

            data_display.dataframe(df)
            stframe.image(frame, channels="RGB")
        
