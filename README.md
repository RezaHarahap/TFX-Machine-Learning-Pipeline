# TFX Machine Learning Pipeline — Breast Cancer Classification

Repository ini berisi proyek **Pengembangan Machine Learning Pipeline** menggunakan **TensorFlow Extended (TFX)** untuk membangun pipeline klasifikasi kanker payudara yang dapat direproduksi dari tahap ingestion data hingga model serving.

**Author:** Muhammad Reza Pahlevi Harahap  
**Username Dicoding:** `reza_harahap`

## Tentang Proyek

Proyek menggunakan **Breast Cancer Wisconsin Diagnostic Dataset** dari `sklearn.datasets.load_breast_cancer`. Dataset memiliki **569 observasi**, **30 fitur numerik**, dan target biner:

- `0` = malignant (ganas)
- `1` = benign (jinak)

Model dibuat untuk tujuan pembelajaran dan **bukan alat diagnosis medis**.

## Alur Pipeline TFX

Pipeline dijalankan menggunakan `InteractiveContext` dengan komponen:

1. `CsvExampleGen` — membaca dataset CSV dan membagi data train/eval.
2. `StatisticsGen` — menghasilkan statistik dataset.
3. `SchemaGen` — membuat schema data.
4. `ExampleValidator` — memeriksa anomali data.
5. `Transform` — melakukan normalisasi fitur numerik dengan z-score.
6. `Tuner` — mencari hyperparameter terbaik menggunakan Random Search.
7. `Trainer` — melatih model Keras menggunakan hyperparameter terbaik.
8. `Resolver` — mencari model blessed sebelumnya sebagai baseline.
9. `Evaluator` — mengevaluasi model menggunakan TensorFlow Model Analysis.
10. `Pusher` — meneruskan model yang lolos evaluasi ke direktori serving.

## Arsitektur Model

Model menggunakan Keras Functional API dengan struktur utama:

- input untuk 30 fitur hasil transformasi;
- concatenation;
- Dense ReLU;
- Dropout;
- Dense ReLU;
- output Dense 1 dengan aktivasi sigmoid.

Model dikompilasi menggunakan Adam, binary cross-entropy, serta metric Binary Accuracy, AUC, Precision, dan Recall.

## Hasil Evaluasi

Pipeline telah dijalankan penuh menggunakan **TFX 1.16.0** dan **TensorFlow 2.16.1**.

Hasil TensorFlow Model Analysis pada split evaluasi:

| Metric | Hasil |
|---|---:|
| AUC | **0.9942** |
| Binary Accuracy | **0.9737** |
| Validation | **Lulus** |
| Model Blessing | **BLESSED** |

Target minimum evaluasi adalah AUC **0.80** dan Binary Accuracy **0.80**, sehingga model memenuhi kriteria untuk diteruskan oleh `Pusher` ke `serving_model`.

Hyperparameter terbaik yang diperoleh Tuner:

- hidden layer pertama: **32 unit**;
- dropout rate: **0.1**;
- hidden layer kedua: **16 unit**;
- learning rate: **0.001**.

## Struktur Proyek

```text
TFX-Machine-Learning-Pipeline/
├── data/
│   ├── README.md
│   └── csv/
│       └── breast_cancer.csv
├── modules/
│   ├── transform_module.py
│   ├── tuner_module.py
│   └── trainer_module.py
├── reza_harahap-pipeline.ipynb
├── reza_harahap-pipeline/
│   ├── CsvExampleGen/
│   ├── StatisticsGen/
│   ├── SchemaGen/
│   ├── ExampleValidator/
│   ├── Transform/
│   ├── Tuner/
│   ├── Trainer/
│   ├── Evaluator/
│   └── Pusher/
├── serving_model/
├── CARA_MENJALANKAN_COLAB.md
├── CHECKLIST_SEBELUM_SUBMIT.md
├── requirements.txt
└── README.md
```

Direktori pipeline pada repository berisi metadata dan artefak hasil eksekusi komponen TFX yang tersimpan dari proses pipeline.

## Cara Menjalankan

1. Ikuti panduan rinci pada `CARA_MENJALANKAN_COLAB.md`.
2. Buka `reza_harahap-pipeline.ipynb` di Google Colab.
3. Gunakan runtime Python yang kompatibel dengan dependency pada `requirements.txt`.
4. Jalankan instalasi dependency dan restart runtime jika diperlukan.
5. Jalankan seluruh cell notebook dari awal hingga akhir.
6. Pastikan setiap komponen pipeline selesai tanpa error.
7. Periksa hasil `Evaluator` dan status model blessing.
8. Model yang lolos akan tersedia pada direktori `serving_model`.

## Teknologi

- Python
- TensorFlow
- TensorFlow Extended (TFX)
- TensorFlow Transform
- TensorFlow Model Analysis
- Keras
- Keras Tuner
- scikit-learn
- Google Colab / Jupyter Notebook

## Tujuan Pembelajaran

Project ini menunjukkan implementasi end-to-end machine learning pipeline yang mencakup validasi data, preprocessing, hyperparameter tuning, training, evaluasi, model blessing, dan model serving secara terstruktur menggunakan TFX.

---

**Muhammad Reza Pahlevi Harahap**  
GitHub: [RezaHarahap](https://github.com/RezaHarahap)
