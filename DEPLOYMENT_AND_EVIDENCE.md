# Deployment & Evidence Final

Semua bukti di paket ini berasal dari runtime nyata.

## TFX pipeline
- Environment eksekusi: image resmi `tensorflow/tfx:1.15.1` / Python 3.10.
- `BeamDagRunner` benar-benar dieksekusi dan output disimpan pada `reza_harahap.ipynb` serta `reza_harahap-pipeline.ipynb`.
- Artifact final: CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Tuner, Trainer, Resolver, Evaluator, Pusher.
- Best hyperparameters: learning_rate=0.01, hidden_units=32, dropout=0.30.
- Evaluator: BinaryAccuracy=0.9705882353, AUC=0.9944022391, validation_ok=True, BLESSED=True.

## Railway / TensorFlow Serving
- URL: `https://mlops-dicoding-reza-production.up.railway.app`
- Model: `breast_cancer_model`, version 2, AVAILABLE.
- Serving input: `examples`, `DT_STRING`, tensor `serving_default_examples:0`.
- Prediction payload: serialized `tf.Example` encoded Base64.
- Prediction result: HTTP 200, `predictions: [[1.2218851e-07]]`.
- Full executable evidence: `reza_harahap-testing.ipynb` and `railway-test-result.txt`.

## Monitoring
- Prometheus public TensorFlow Serving target is shown as 1/1 UP in `evidence/reza_harahap-prometheus-cloud-targets.png`.
- Additional live metric evidence is included in `evidence/reza_harahap-prometheus-graph.png` and `evidence/prometheus-live-timeseries.csv`.

## Screenshots
- `evidence/reza_harahap-deployment.png`: latest Railway deployment ACTIVE.
- `evidence/reza_harahap-model-metadata-1.png` and `-2.png`: deployed model metadata.
- `evidence/reza_harahap-tfx-actions-success.png`: successful official TFX GitHub Actions execution.
- `evidence/reza_harahap-pylint.png`: pylint evidence.
