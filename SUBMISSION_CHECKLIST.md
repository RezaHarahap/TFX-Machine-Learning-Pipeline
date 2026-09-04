# Submission Checklist — Final Reviewer Fix

## Kriteria pipeline TFX
- [x] `pipeline.Pipeline` berisi CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Tuner, Trainer, Resolver, Evaluator, Pusher.
- [x] `BeamDagRunner` benar-benar dijalankan.
- [x] Output eksekusi Beam tersimpan di notebook utama.
- [x] Artifact TFX nyata disertakan.
- [x] Tuner artifact dan arsitektur README konsisten: LR 0.01, hidden_units 32, dropout 0.30; Dense32 → Dropout0.30 → Dense16.
- [x] Evaluator menggunakan BinaryAccuracy dan AUC; final BLESSED=True.

## Deployment cloud
- [x] Railway deployment ACTIVE / successful.
- [x] Docker image menggunakan SavedModel TFX Pusher `serving_model/2`.
- [x] Metadata produksi: `examples` / `DT_STRING`, bukan `features[-1,30]`.
- [x] Testing menggunakan serialized `tf.Example` Base64.
- [x] Prediction publik berhasil HTTP 200 dan menghasilkan `predictions`.
- [x] Notebook testing menyimpan output runtime nyata.

## Monitoring dan dokumentasi
- [x] Prometheus target cloud TensorFlow Serving 1/1 UP.
- [x] Prometheus metrics config disertakan.
- [x] Screenshot deployment terbaru disertakan.
- [x] README disinkronkan dengan Tuner, Evaluator, Pusher, dan kontrak serving final.
- [x] Pylint/syntax evidence tersedia.
