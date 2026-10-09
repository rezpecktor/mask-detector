# Mask Detector

Aplikasi computer vision sederhana untuk mengklasifikasikan kondisi bermasker atau tanpa masker dari kamera. Proyek ini dibuat sebagai proyek pembelajaran dan portofolio; hasilnya belum divalidasi untuk penggunaan keselamatan atau produksi.

## Fitur

- Klasifikasi frame kamera menjadi `Mask` atau `No Mask`.
- Prediksi dirata-ratakan dari beberapa frame untuk mengurangi perubahan label yang cepat.
- Memutar suara saat label berubah.
- Skrip terpisah untuk melatih model dan memeriksa kamera.

## Struktur proyek

```text
.
├── main.py
├── train_model.py
├── test_camera.py
├── requirements.txt
├── mask_on.wav
├── mask_off.wav
└── model/
    └── mask_detector_model.h5
```

Dataset tidak disertakan dalam Git karena sumber dan izin redistribusinya perlu dipastikan terlebih dahulu. Folder lokal `dataset/` diabaikan oleh Git. Untuk melatih ulang model, siapkan dataset dengan struktur berikut setelah memastikan dataset tersebut boleh digunakan:

```text
dataset/
├── with_mask/
└── without_mask/
```

## Persyaratan

- Python 3.12 (versi yang digunakan saat proyek ini disiapkan: 3.12.4)
- Kamera yang dapat diakses oleh OpenCV
- Dependensi pada `requirements.txt`

## Instalasi dan menjalankan

Buat dan aktifkan virtual environment, lalu pasang dependensi:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jalankan detektor dari direktori utama proyek:

```powershell
python main.py
```

Tekan `q` pada jendela kamera untuk keluar. Skrip membuka kamera indeks `0`; ubah `camera_index` di `main.py` jika perangkat kamera Anda menggunakan indeks lain.

Untuk memeriksa kamera tanpa memuat model:

```powershell
python test_camera.py
```

Untuk melatih ulang model (dataset harus sudah tersedia secara lokal):

```powershell
python train_model.py
```

Model hasil training disimpan ke `model/mask_detector_model.h5`. Training memakai bobot awal ImageNet dari MobileNetV2; unduhan bobot mungkin diperlukan saat pertama kali dijalankan.

## Catatan dan batasan

- Aplikasi mengklasifikasikan seluruh frame kamera dan belum mendeteksi atau memotong wajah secara terpisah.
- Preprocessing MobileNetV2 dan kualitas model perlu dievaluasi lebih lanjut sebelum menyatakan akurasi atau menggunakannya untuk keputusan penting.
- Hasil dapat dipengaruhi pencahayaan, posisi wajah, kamera, dan data training.
- Dataset, bobot pretrained, serta audio dapat memiliki lisensi dan ketentuan atribusi tersendiri. Periksa ketentuannya sebelum mendistribusikan proyek.

## Lisensi

Lisensi proyek belum ditentukan. Tambahkan berkas lisensi setelah memastikan hak atas kode dan aset yang disertakan.
