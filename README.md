# Submission 1: Breast Cancer Classification MLOps

**Nama:** Muhammad Reza Pahlevi Harahap  
**Username Dicoding:** `reza_harahap`

| | Deskripsi |
| ----------- | ----------- |
| Dataset | **Breast Cancer Wisconsin (Diagnostic)**. Dataset submission berada pada `data/breast_cancer.csv` dan terdiri dari **569 observasi**, **30 fitur numerik**, serta satu label biner. Distribusi label pada berkas yang dipakai pipeline adalah 212 sampel label 0 dan 357 sampel label 1. |
| Masalah | Proyek menyelesaikan klasifikasi biner karakteristik tumor payudara berdasarkan fitur numerik hasil pengukuran inti sel. Sistem tidak hanya melatih model, tetapi juga membangun pipeline data/model yang reproducible, menyiapkan model untuk serving, melakukan deployment, serta memonitor layanan produksi. |
| Solusi machine learning | Solusi dibangun sebagai **TensorFlow Extended (TFX) pipeline** menggunakan `BeamDagRunner`. Komponen yang digunakan adalah `CsvExampleGen`, `StatisticsGen`, `SchemaGen`, `ExampleValidator`, `Transform`, `Tuner`, `Trainer`, `Resolver`, `Evaluator`, dan `Pusher`. Model yang lolos evaluasi dipush ke `serving_model/` sebagai TensorFlow SavedModel. |
| Metode pengolahan | `CsvExampleGen` membagi input menjadi contoh train/eval dan menyimpannya sebagai TFRecord. `StatisticsGen` menghasilkan statistik fitur, `SchemaGen` membuat schema, dan `ExampleValidator` memeriksa anomali. Pada `Transform`, seluruh 30 fitur numerik distandardisasi menggunakan z-score (`tft.scale_to_z_score`), sedangkan label dipertahankan sebagai integer untuk klasifikasi biner. |
| Arsitektur model | `Tuner` pada eksekusi final memilih **learning_rate = 0.01**, **hidden_units = 32**, dan **dropout = 0.30** (`best_hyperparameters.txt`). Karena Trainer membangun layer kedua dengan `max(hidden_units // 2, 8)`, arsitektur model yang benar-benar dilatih adalah **Dense(32, ReLU) → Dropout(0.30) → Dense(16, ReLU) → Dense(1, sigmoid)**. Optimizer menggunakan Adam dengan learning rate 0.01 dan loss `binary_crossentropy`. |
| Metrik evaluasi | `Evaluator` menggunakan TensorFlow Model Analysis dengan **BinaryAccuracy** dan **AUC**, masing-masing threshold minimal **0.80** sesuai `pipeline.py`. Pada eksekusi final, overall slice menghasilkan **BinaryAccuracy = 0.9705882353** dan **AUC = 0.9944022391**; `validation_ok=True` dan model menghasilkan artifact **BLESSED**. |
| Performa model | Model final lolos Evaluator (**BLESSED=True**) dan diteruskan ke `Pusher`. SavedModel Pusher berada di `serving_model/2/`. Deployment Railway aktif melayani model version **2** dengan input signature **`examples: DT_STRING`**. Pengujian publik menggunakan serialized `tf.Example` Base64 menghasilkan **HTTP 200** dan response `predictions: [[1.2218851e-07]]`; output lengkap tersimpan di `reza_harahap-testing.ipynb` dan `railway-test-result.txt`. |
| Opsi deployment | Model dipaketkan menggunakan **TensorFlow Serving 2.15.1** di dalam Docker container dan dideploy ke **Railway**. REST API TensorFlow Serving diekspos melalui HTTPS. Prometheus dan Grafana disiapkan dalam image yang sama untuk menghindari kebutuhan service cloud tambahan. |
| Web app | **Model status:** [breast_cancer_model](https://mlops-dicoding-reza-production.up.railway.app/v1/models/breast_cancer_model)  •  **Metadata:** [serving metadata](https://mlops-dicoding-reza-production.up.railway.app/v1/models/breast_cancer_model/metadata)  •  endpoint prediksi: `POST https://mlops-dicoding-reza-production.up.railway.app/v1/models/breast_cancer_model:predict` |
| Monitoring | TensorFlow Serving mengekspor metric pada `/monitoring/prometheus/metrics`. Prometheus dikonfigurasi untuk melakukan scrape melalui **public cloud target** `https://mlops-dicoding-reza-production.up.railway.app/monitoring/prometheus/metrics`, bukan hostname internal/localhost. UI Prometheus dapat diakses pada [Targets](https://mlops-dicoding-reza-production-992b.up.railway.app/targets). Konfigurasi juga memuat `evaluation_interval`, `external_labels`, dan job-level `scrape_interval`. |

## Struktur Pipeline TFX

1. **CsvExampleGen** — membaca `data/breast_cancer.csv` dan menghasilkan TFRecord train/eval.
2. **StatisticsGen** — menghasilkan statistics protobuf dari examples.
3. **SchemaGen** — menghasilkan schema protobuf text.
4. **ExampleValidator** — menghasilkan hasil validasi/anomaly artifact.
5. **Transform** — menghasilkan transform graph serta transformed examples.
6. **Tuner** — mencari hyperparameter model.
7. **Trainer** — melatih dan mengekspor TensorFlow SavedModel.
8. **Resolver** — mengambil model terakhir yang telah diberkati sebagai baseline bila tersedia.
9. **Evaluator** — mengevaluasi model dengan Binary Accuracy dan AUC.
10. **Pusher** — menyalin model yang diberkati ke `serving_model/` untuk production serving.

## Menjalankan Proyek

Gunakan **Python 3.10** untuk kompatibilitas TFX 1.15.x.

```bash
pip install -r requirements.txt
python pipeline.py
```

Notebook `reza_harahap-pipeline.ipynb` mendokumentasikan inisialisasi komponen, pembangunan pipeline, eksekusi `BeamDagRunner`, dan verifikasi artifact. Notebook `reza_harahap-testing.ipynb` melakukan request langsung ke URL Railway sebenarnya dan harus disertakan bersama output hasil eksekusinya.

## Deployment dan Monitoring

Docker image menjalankan TensorFlow Serving pada port 8501, Prometheus pada 9090, dan Grafana pada 3000. Bukti deployment serta monitoring yang digunakan untuk review terdapat pada folder `evidence/cloud/` dan berasal dari endpoint cloud aktual. Prometheus submission config terdapat pada `monitoring/prometheus.yml`.


## Bukti Eksekusi Final

- `reza_harahap.ipynb` dan `reza_harahap-pipeline.ipynb`: notebook hasil eksekusi nyata pada image resmi `tensorflow/tfx:1.15.1`; memuat output `BeamDagRunner`, 73 artifact TFX, Tuner, Evaluator `BLESSED`, dan Pusher.
- `reza_harahap-testing.ipynb`: notebook testing Railway yang sudah dieksekusi; status model HTTP 200, metadata `examples / DT_STRING`, dan prediction HTTP 200.
- `railway-test-result.txt`: salinan teks output pengujian cloud otomatis.
- `evidence/reza_harahap-deployment.png`: deployment Railway terbaru berstatus ACTIVE / successful.
- `evidence/reza_harahap-prometheus-cloud-targets.png`: target TensorFlow Serving cloud terlihat 1/1 UP.
- `serving_model/2/`: SavedModel yang berasal dari komponen TFX `Pusher` dan digunakan oleh TensorFlow Serving.
