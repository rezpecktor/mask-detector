import cv2

# Pilih indeks kamera
camera_index = 0  # Ganti dengan indeks kamera eksternal Anda
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Tidak dapat membuka kamera dengan indeks {camera_index}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tampilkan frame dari kamera eksternal
    cv2.imshow('Kamera Eksternal', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
