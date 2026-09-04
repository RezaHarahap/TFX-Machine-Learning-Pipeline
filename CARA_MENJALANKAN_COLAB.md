# Cara Menjalankan Proyek di Google Colab

## A. Persiapan

1. Ekstrak ZIP proyek di laptop.
2. Unggah folder `reza_harahap_tfx_bintang3` ke Google Drive.
3. Buka Google Colab dan pilih runtime Python yang kompatibel dengan TFX 1.16.
4. Hubungkan Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

5. Pindah ke folder proyek. Sesuaikan path jika folder Drive berbeda:

```python
%cd /content/drive/MyDrive/reza_harahap_tfx_bintang3
```

6. Buka `reza_harahap-pipeline.ipynb` dan pastikan current working directory
   adalah folder proyek di atas.

## B. Eksekusi

1. Jalankan cell instalasi.
2. Jika Colab meminta restart runtime, lakukan restart.
3. Hubungkan Drive dan jalankan `%cd` kembali.
4. Pilih **Runtime > Run all**.
5. Jangan menutup tab selama Tuner, Trainer, Evaluator, dan Pusher berjalan.

## C. Pemeriksaan sebelum diunduh

Pastikan:

- notebook tidak memiliki cell error;
- output setiap cell tersimpan;
- folder `reza_harahap-pipeline` berisi artefak komponen;
- folder `serving_model` berisi SavedModel setelah model memperoleh blessing;
- output Evaluator memperlihatkan model memperoleh blessing;
- README sudah diisi dengan performa aktual, bukan angka asumsi;
- `breast_cancer.csv` sudah berada di folder `data/csv`.

## D. Membuat ZIP submission

1. Unduh folder proyek hasil eksekusi dari Google Drive.
2. Buka notebook lokal dan pastikan output masih terlihat.
3. Hapus folder `__pycache__` jika ada.
4. Kompres folder `reza_harahap_tfx_bintang3` menjadi satu ZIP.
5. Pastikan tidak ada ZIP kedua di dalam ZIP submission.

## Catatan penilaian

Proyek dirancang untuk asumsi **bintang 3** karena seluruh komponen wajib ada dan
menerapkan satu saran tambahan berupa Tuner. Proyek tidak menerapkan TensorFlow
Serving melalui Docker maupun notebook prediction request. Kelulusan dan nilai
tetap ditentukan reviewer serta bergantung pada keberhasilan eksekusi dan
kelengkapan artefak yang dikirim.
