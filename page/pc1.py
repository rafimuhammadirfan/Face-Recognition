import cv2
import pandas as pd
import numpy as np
import streamlit as st
import time
from ultralytics import YOLO
from collections import defaultdict
from scipy.spatial import distance
from datetime import datetime

st.markdown(
    """
    <div style='text-align: center;'>
        <h1>PEOPLE COUNTING</h1>
        <p>Pilih metode people counting yang akan digunakan dan nyalakan kamera untuk memulai fitur people counting.</p>
    </div>
    """,
    unsafe_allow_html=True
)
tab1, tab2, tab3 = st.tabs(["Real-time People Counting", "Door People Counting", "Setting"])

# Load YOLO model
model = YOLO('data/yolov8l.engine')

# Define buffer to store the detection results
if "count_data" not in st.session_state:
    st.session_state.count_data = pd.DataFrame(columns=["Waktu", "Jumlah", "Tanggal"])

# Pilihan kamera
with tab3:
    camera_options = ["cam 1", "cam 2", "cam 3"]
    option = st.selectbox(
        "Pilih Kamera:",
        camera_options,
        index=0,
        placeholder="Pilih kamera untuk digunakan...",
    )
    camera_index = camera_options.index(option)

    cam = st.text_input("Masukan URL camera")

# Real-time People Counting
with tab1:
    def process_frame(frame):
        frame = cv2.resize(frame, (1020, 900))
        results = model.predict(frame, conf=0.5, iou=0.5, augment=False, stream=True, device=0)
        
        person_count = 0  # Inisialisasi counter untuk orang
        for result in results:
            boxes = result.boxes  # Get the bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates
                score = box.conf[0]  # Confidence score
                class_id = int(box.cls[0])  # Class ID

                if class_id == 0:  # Assuming class_id 0 is for 'person'
                    person_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tambahkan jumlah orang yang terdeteksi ke dalam frame
        cv2.putText(frame, f'Persons detected: {person_count}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, person_count
    st.write("Pilih camera yang akan digunakan pada menu setting dan nyalakan camera untuk memulai fitur real-time people counting.")
    # st.markdown(
    #     """
    #     <div style='text-align: center;'>
    #          <p>Pilih camera yang akan digunakan pada menu setting dan nyalakan camera untuk memulai fitur real-time people counting.</p>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    stframe = st.empty()
    frame_window = st.image([])

    if cam:
        cap = cv2.VideoCapture(cam)
    else:
        cap = cv2.VideoCapture(int(f"{camera_index}"))

    on = st.toggle("Camera", key="webcam1")

    # Buffer untuk menyimpan jumlah orang yang terdeteksi setiap menit
    person_count_buffer = []
    start_time = time.time()

    if on:
        prev_time = 0  # Inisialisasi waktu sebelumnya untuk perhitungan FPS
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, person_count = process_frame(frame)
            
            # Hitung FPS
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = 1 / time_diff
            prev_time = curr_time
            
            # Tambahkan jumlah orang yang terdeteksi ke buffer
            person_count_buffer.append(person_count)
            
            # Hitung rata-rata jumlah orang per menit
            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                avg_person_count = sum(person_count_buffer) / len(person_count_buffer)
                st.write(f'Average persons detected per minute: {round(avg_person_count)}')
                person_count_buffer = []  # Reset buffer
                start_time = time.time()  # Reset time

            # Simpan data ke dalam DataFrame
                current_time_str = time.strftime("%H:%M:%S", time.localtime())
                current_date = time.strftime("%Y-%m-%d", time.localtime())
                new_row = pd.DataFrame([{"Waktu": current_time_str, "Jumlah": person_count, "Tanggal": current_date}])
                st.session_state.count_data = pd.concat([st.session_state.count_data, new_row], ignore_index=True)

            # Tampilkan waktu real-time dan FPS di frame
                real_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(frame, f'Time: {real_time}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'FPS: {fps:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            stframe.image(frame, channels="RGB")
            
        cap.release()
        
    st.dataframe(st.session_state.count_data, width=2000)

    # Tambahkan tombol untuk mengunduh CSV
    csv = st.session_state.count_data.to_csv(index=False)
    a,b,c = st.columns(3, vertical_alignment="top", gap="small")
    b.download_button(
        label=":material/download: Download CSV",
        data=csv,
        file_name='person_count_data.csv',
        mime='text/csv'
    )

# Door People Counting
with tab2:
    area1 = [(500, 10), (500, 600), (450, 600), (450, 10)]
    area2 = [(400, 10), (400, 600), (350, 600), (350, 10)]
    with open ("data/coco.txt", "r")as my_file:
        class_list=my_file.read().split("\n")
    # Variables for person counting and tracking
    person_count = 0
    previous_positions = {}
    delay_counter = defaultdict(lambda: 0)
    delay_threshold = 15  # Delay before detecting the same person again

    person_count_buffer = []
    start_time = time.time()

    stframe = st.empty()

    def process_frame(frame, person_count, previous_positions, delay_counter):
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data.cpu().numpy()
        px = pd.DataFrame(a).astype("float")
        
        current_positions = {}

        for index, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            d = int(row[5])  # Class index
            c = class_list[d]

            if 'person' in c:
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                current_positions[index] = (centroid_x, centroid_y)

                if previous_positions:
                    closest_idx = min(previous_positions, key=lambda i: distance.euclidean(previous_positions[i], (centroid_x, centroid_y)))
                    prev_x, prev_y = previous_positions[closest_idx]

                    if delay_counter[closest_idx] == 0:
                        if prev_x > 400 and centroid_x < 450:  # Movement from area 2 to area 1
                            person_count += 1
                            delay_counter[closest_idx] = delay_threshold
                        elif prev_x < 450 and centroid_x > 400:  # Movement from area 1 to area 2
                            person_count -= 1
                            delay_counter[closest_idx] = delay_threshold

        previous_positions = current_positions

        for k in delay_counter:
            if delay_counter[k] > 0:
                delay_counter[k] -= 1

        cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
        cv2.putText(frame, str('1'), (500, 400), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
        cv2.putText(frame, str('2'), (400, 400), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

        cv2.putText(frame, f'Persons count: {person_count}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, person_count, previous_positions, delay_counter

    # Streamlit UI
    if cam:
        cap = cv2.VideoCapture(cam)
    else:
        cap = cv2.VideoCapture(int(f"{camera_index}"))

    on = st.toggle("Camera", key="webcam2")

    if on:
        prev_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame, person_count, previous_positions, delay_counter = process_frame(frame, person_count, previous_positions, delay_counter)

            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = 1 / time_diff
            prev_time = curr_time

            person_count_buffer.append(person_count)

            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                avg_person_count = sum(person_count_buffer) / len(person_count_buffer)
                st.write(f'Average persons detected per minute: {round(avg_person_count)}')
                person_count_buffer = []
                start_time = time.time()

            # Simpan data ke dalam DataFrame
            current_time_str = time.strftime("%H:%M:%S", time.localtime())
            current_date = time.strftime("%Y-%m-%d", time.localtime())
            new_row = pd.DataFrame([{"Waktu": current_time_str, "Jumlah": person_count, "Tanggal": current_date}])
            st.session_state.count_data = pd.concat([st.session_state.count_data, new_row], ignore_index=True)

            real_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, f'Time: {real_time}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'FPS: {fps:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            stframe.image(frame, channels="RGB")

        cap.release()

# Tampilkan DataFrame hasil deteksi
