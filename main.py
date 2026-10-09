import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from pygame import mixer
from collections import deque

# Inisialisasi pygame mixer untuk memutar suara
mixer.init()

# Fungsi untuk memutar suara
def play_sound(file_path):
    mixer.music.load(file_path)
    mixer.music.play()
    while mixer.music.get_busy():  # Tunggu hingga audio selesai diputar
        pass

# Load model
model = load_model('./model/mask_detector_model.h5')

# Label encoder
labels = ['Mask', 'No Mask']

# Pilih kamera (0 untuk kamera laptop, 1 untuk kamera eksternal)
camera_index = 0  # Ganti ke 1 untuk menggunakan kamera eksternal
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Tidak dapat membuka kamera dengan indeks {camera_index}. Pastikan kamera tersambung.")
    exit()

# Atur resolusi kamera (opsional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variabel untuk smoothing prediksi
buffer_size = 5
predictions_buffer = deque(maxlen=buffer_size)
last_label = None

# Threshold probabilitas
threshold = 0.8

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Balik frame untuk menghindari efek mirror
    frame = cv2.flip(frame, 1)

    # Preprocess frame
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0]
    predictions_buffer.append(prediction)

    # Rata-rata prediksi untuk smoothing
    avg_prediction = np.mean(predictions_buffer, axis=0)

    # Tentukan label berdasarkan probabilitas dan threshold
    if avg_prediction[1] >= threshold:  # Jika probabilitas No Mask >= threshold
        label = 'No Mask'
    else:
        label = 'Mask'

    # Debug probabilitas dan label
    print(f"Probabilitas Mask: {avg_prediction[0]:.2f}, Probabilitas No Mask: {avg_prediction[1]:.2f}, Label: {label}")

    # Warna
    text_color = (0, 0, 0)  # Hitam untuk teks
    background_color = (255, 255, 255)  # Putih untuk latar belakang teks
    border_color = (0, 0, 0)  # Hitam untuk garis tepi kotak

    # Hitung ukuran teks
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
    text_width, text_height = text_size

    # Koordinat dan ukuran kotak dinamis
    top_left = (20, 50)  # Koordinat atas-kiri
    bottom_right = (20 + text_width + 20, 50 + text_height + 20)  # Tambahkan padding

    # Gambar kotak putih sebagai background teks
    cv2.rectangle(frame, top_left, bottom_right, background_color, -1)  # -1 untuk kotak solid

    # Gambar garis tepi kotak hitam
    cv2.rectangle(frame, top_left, bottom_right, border_color, 2)  # 2 untuk ketebalan garis tepi

    # Tambahkan teks di atas latar belakang
    text_position = (top_left[0] + 10, top_left[1] + text_height + 5)  # Tambahkan padding untuk teks
    cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 4)

    # Mainkan suara jika status berubah
    if label != last_label:
        last_label = label
        if label == 'Mask':
            play_sound(r'./mask_on.wav')  # Suara ketika memakai masker
        elif label == 'No Mask':
            play_sound(r'./mask_off.wav')  # Suara ketika tidak memakai masker

    # Tampilkan frame
    cv2.imshow('Mask Detector', frame)

    # Break dengan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
