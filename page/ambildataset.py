import cv2
import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from keras_facenet import FaceNet
from torchvision import transforms
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import altair as alt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

tab1, tab2 = st.tabs(["Ambil Dataset", "Training"])

# Tab 1 - Ambil Dataset
with tab1:
    st.markdown("<div style='text-align: center;'><h1>AMBIL DATASET</h1></div>", unsafe_allow_html=True)

    depan = st.text_input("Nama depan:")
    belakang = st.text_input("Nama belakang:")
    my_name = depan + " " + belakang

    if 'webcam_stream' not in st.session_state:
        st.session_state.webcam_stream = None

    # Sidebar camera selection
    camera_options = ["cam 1", "cam 2", "cam 3"]
    option = st.sidebar.selectbox("Pilih Kamera:", camera_options, index=0, placeholder="Pilih kamera untuk digunakan...")
    camera_index = camera_options.index(option)

    # Check if the name fields are empty
    if depan.strip() == "" or belakang.strip() == "":
        st.warning("Harap isi nama sebelum mengambil gambar!")
        toggle = True
    else:
        toggle = False

    on = st.toggle("🎦Kamera", key="webcam_checkbox", disabled=toggle)

    if on:
        if st.session_state.webcam_stream is None:
            st.session_state.webcam_stream = cv2.VideoCapture(1)

        stframe = st.empty()
        bar = st.progress(0)
        q, w, e = st.columns(3)
        dataset_folder = st.text_input("path/to/new/folder")
        take_picture1 = q.button("Ambil Gambar tampak depan", key="take_picture1")
        take_picture2 = w.button("Ambil Gambar tampak samping kanan", key="take_picture2")
        take_picture3 = e.button("Ambil Gambar tampak samping Kiri", key="take_picture3")

        num_sample1 = 11
        i1 = 1
        num_sample2 = 21
        i2 = 11
        num_sample3 = 31
        i3 = 21

        while on:
            ret, frame = st.session_state.webcam_stream.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                continue

            if take_picture1 and i1 < num_sample1:
                ret, frame = st.session_state.webcam_stream.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                user_folder = os.path.join(dataset_folder, my_name)
                os.makedirs(user_folder, exist_ok=True)

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resize = cv2.resize(frame_rgb, (250, 250))
                    cv2.imwrite(f"{user_folder}/{my_name}_{i1:04d}.jpg", frame_resize)
                    i1 += 1
                    bar.progress(i1 / num_sample1)
                    time.sleep(0.5)
                    if i1 == num_sample1:
                        st.success('Dataset tampak depan telah diambil', icon="✅")
                        bar.empty()

            if take_picture2 and i2 < num_sample2:
                ret, frame = st.session_state.webcam_stream.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                user_folder = os.path.join(dataset_folder, my_name)
                os.makedirs(user_folder, exist_ok=True)

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resize = cv2.resize(frame_rgb, (250, 250))
                    cv2.imwrite(f"{user_folder}/{my_name}_{i2:04d}.jpg", frame_resize)
                    i2 += 1
                    bar.progress(i2 / num_sample2)
                    if i2 == num_sample2:
                        st.success('Dataset tampak samping kanan telah diambil', icon="✅")
                        bar.empty()

            if take_picture3 and i3 < num_sample3:
                ret, frame = st.session_state.webcam_stream.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                user_folder = os.path.join(dataset_folder, my_name)
                os.makedirs(user_folder, exist_ok=True)

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resize = cv2.resize(frame_rgb, (250, 250))
                    cv2.imwrite(f"{user_folder}/{my_name}_{i3:04d}.jpg", frame_resize)
                    i3 += 1
                    bar.progress(i3 / num_sample3)
                    if i3 == num_sample3:
                        st.success('Dataset tampak samping kiri telah diambil', icon="✅")
                        bar.empty()

            stframe.image(frame, channels="RGB")

    else:
        if st.session_state.webcam_stream is not None:
            st.session_state.webcam_stream.release()
            st.session_state.webcam_stream = None


class FACE:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.model = YOLO('data/best1.engine')

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomResizedCrop(size=self.target_size, scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])

    def extract_face(self, filename):
        img = cv2.imread(filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.predict(img_rgb)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face = img_rgb[y1:y2, x1:x2]
                    face_resized = cv2.resize(face, self.target_size)
                    return face_resized
        return None

    def augment_face(self, face_image):
        face_pil = transforms.ToPILImage()(face_image)
        augmented = self.augmentation(face_pil)
        augmented = np.array(augmented)
        return augmented

    def load_faces(self, dir):
        FACES = []
        for img_name in os.listdir(dir):
            try:
                path = os.path.join(dir, img_name)
                single_face = self.extract_face(path)
                if single_face is not None:
                    augmented_face = self.augment_face(single_face)
                    FACES.append(augmented_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        initial_class_counts = {}
        df_initial = pd.DataFrame(columns=['Class', 'Faces'])

        # First, count the number of faces in each class
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)
            LABEL = [sub_dir for _ in range(len(FACES))]
            initial_class_counts[sub_dir] = len(FACES)

            # Add to the initial DataFrame
            df_initial = pd.concat([df_initial, pd.DataFrame({'Class': [sub_dir], 'Faces': [len(FACES)]})], ignore_index=True)

            self.X.extend(FACES)
            self.Y.extend(LABEL)

        # Determine the minimum number of faces in any class
        min_faces = min(initial_class_counts.values())

        # Downsampling the majority classes
        df_balanced = pd.DataFrame(columns=['Class', 'Faces'])

        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)

            # If the class has more than the minimum faces, downsample
            if len(FACES) > min_faces:
                FACES = FACES[:min_faces]

            LABEL = [sub_dir for _ in range(len(FACES))]
            self.X.extend(FACES)
            self.Y.extend(LABEL)

            # Add to the balanced DataFrame
            df_balanced = pd.concat([df_balanced, pd.DataFrame({'Class': [sub_dir], 'Faces': [len(FACES)]})], ignore_index=True)

        # Visualization of the class distribution before and after downsampling
        st.sidebar.subheader("Class Distribution (Before Downsampling)")
        st.sidebar.dataframe(df_initial)
        chart_initial = alt.Chart(df_initial).mark_bar().encode(
            x='Class', 
            y='Faces', 
            color='Class'
        ).properties(width=800, height=400)
        st.altair_chart(chart_initial)

        st.sidebar.subheader("Class Distribution (After Downsampling)")
        st.sidebar.dataframe(df_balanced)
        chart_balanced = alt.Chart(df_balanced).mark_bar().encode(
            x='Class', 
            y='Faces', 
            color='Class'
        ).properties(width=800, height=400)
        st.altair_chart(chart_balanced)

        return np.asarray(self.X), np.asarray(self.Y)


# Tab 2 - Dataset & Training
with tab2:
    q, w = st.columns(2, vertical_alignment="top", gap="small")
    st.markdown("<div style='text-align: center;'><h1>DATASET</h1><p> Berikut merupakan kumpulan dataset yang tersimpan:</p><p></p></div>", unsafe_allow_html=True)

    dataset_folder = st.text_input("Masukan path folder dataset:")

    if dataset_folder:
        subfolders = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
        cols = st.columns(4)

        for i, subfolder in enumerate(subfolders):
            images = [f for f in os.listdir(subfolder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            random_image = random.choice(images)
            image_path = os.path.join(subfolder, random_image)

            col_index = i % 4  # Get the column index (0, 1, or 2)
            with cols[col_index]:
                st.write(f"{os.path.basename(subfolder)}")
                img = Image.open(image_path)
                st.image(img, caption=random_image)
    else:
        st.warning("Harap masukkan path folder dataset.")

    # Training
    row1 = st.columns(5, vertical_alignment="top", gap="small")
    row2 = st.columns(3, vertical_alignment="top", gap="small")
    q1, w1, e1, r1, t1 = row1
    y2, u2, i2 = row2
    status = st.empty()

    if e1.button("Training", icon=":material/model_training:"):
        status.status("Loading dataset...")
        faceloading = FACE(dataset_folder)
        X, Y = faceloading.load_classes()
        time.sleep(1)  # Simulate delay for loading dataset

        # Step 2: Initialize FaceNet and calculate embeddings
        status.status("Calculating embeddings...")
        embedder = FaceNet()
        
        def get_embedding(face_img):
            face_img = face_img.astype('float32')
            face_img = np.expand_dims(face_img, axis=0)
            yhat = embedder.embeddings(face_img)
            return yhat[0]
        
        EMBEDDED_X = []
        for img in X:
            EMBEDDED_X.append(get_embedding(img))
        np.savez_compressed('data/face_embeddings1.npz', EMBEDDED_X, Y)

        embedding_data = np.load('data/face_embeddings1.npz')
        EMBEDDED_X = embedding_data['arr_0']
        Y = embedding_data['arr_1']

# Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(EMBEDDED_X)

# Encode the labels as numbers
        label_encoder = LabelEncoder()
        Y_encoded = label_encoder.fit_transform(Y)

# Create a scatter plot for the 2D t-SNE result
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y_encoded, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('2D Visualization of Face Embeddings using t-SNE')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

# Show the plot in Streamlit
        st.pyplot(plt)

        time.sleep(1)

        # Step 3: Encode the labels and split the data
        status.status("Splitting data...")
        label_encoder = LabelEncoder()
        encoded_Y = label_encoder.fit_transform(Y)
        labels = label_encoder.classes_
        df = pd.DataFrame(labels, columns=['labels'])
        df.to_csv('data/classes1.csv', index=False)
        X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, encoded_Y, test_size=0.2, random_state=42)
        time.sleep(1)

        # Step 4: Train the model and save it
        status.status("Training model...")
        model = SVC(kernel='linear',C=1.0, probability=True)
        cv_scores = cross_val_score(model, X_train, Y_train, cv=5)
        model.fit(X_train, Y_train)
        y_train = model.predict(X_train)
        y_pred = model.predict(X_test)


# Save the model
        with open('data/svm_model1.pkl', 'wb') as file:
            pickle.dump(model, file)
        time.sleep(1)

# Generate classification report
        report = classification_report(Y_test, y_pred, target_names=labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report)

# Generate confusion matrix
        conf_matrix = confusion_matrix(Y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        st.pyplot(fig)
        
        status.empty()
        st.success("Training berhasil")