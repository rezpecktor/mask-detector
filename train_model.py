import os
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.model_selection import train_test_split

# Path dataset
DATASET_DIR = './dataset'
CATEGORIES = ['with_mask', 'without_mask']

# Fungsi untuk memuat data
def load_data():
    images = []
    labels = []

    for category in CATEGORIES:
        folder_path = os.path.join(DATASET_DIR, category)
        label = CATEGORIES.index(category)  # 0 untuk 'with_mask', 1 untuk 'without_mask'

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Pastikan file adalah gambar
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                # Baca gambar dan preprocess
                img = cv2.imread(file_path)
                img = cv2.resize(img, (128, 128))  # Resize ke ukuran yang seragam
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

# Preprocess data
images, labels = load_data()

# Debug jumlah gambar dan label
print("Jumlah gambar:", len(images))
print("Jumlah label:", len(labels))
print("Label unik:", set(labels))

# Normalisasi gambar
images = images / 255.0

# Encode label ke dalam format one-hot encoding
labels = to_categorical(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Gunakan MobileNetV2 sebagai model dasar (pre-trained model)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Tambahkan lapisan custom di atasnya
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)  # 2 kelas: Mask dan No Mask

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layer base_model agar tidak dilatih ulang
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Augmentasi data
datagen = ImageDataGenerator(
    rotation_range=30,          # Rotasi hingga 30 derajat
    width_shift_range=0.3,      # Pergeseran horizontal
    height_shift_range=0.3,     # Pergeseran vertikal
    shear_range=0.3,            # Distorsi shearing
    zoom_range=0.3,             # Zoom in/out
    horizontal_flip=True,       # Membalik gambar secara horizontal
    fill_mode='nearest'         # Mengisi piksel kosong setelah augmentasi
)

# Latih model dengan data augmentasi
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_test, y_test),
          epochs=15)  # Latih dengan 15 epoch

# Save model
os.makedirs('./model', exist_ok=True)
model.save('./model/mask_detector_model.h5')

print("Model training selesai dan disimpan ke './model/mask_detector_model.h5'")
