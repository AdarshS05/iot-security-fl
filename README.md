# A Lightweight Adversarial Hardening and Federated Learning based Framework for IoT Malware Detection
Traditional machine learning classifiers are effective in a closed-set scenario, where the samples it tests are similar to that of the training data. However these classifiers are highly susceptible to adversarial perturbations due to inherent weaknesses in deep learning models. Such models also struggle to identify zero day variants without large datasets. Training large datasets, brings a high computational overhead to resource-constrained IoT edge devices. If 
new malware is discovered, centralized retraining is required, which consumes high bandwidth and brings privacy risks when raw samples must be shared across network boundaries. 

## Objectives
1. Design a hybrid IoT malware static analysis framework which integrates structural metadata analysis with assembly level semantic feature analysis 
2. Develop a Genetic Algorithm based framework combining feature-space and binary-level perturbations to improve model resilience  
3. Extend the framework to a federated learning based approach to protect against zero day attacks 
4. Evaluate performance against benchmark detection frameworks and different attack scenarios

Project Repository Structure
------------------
```
src/
  assembly/           assembly utilities / orchestration scripts
  export/             convert/export models to ONNX
  fl/                 federated-learning helpers
    common/           metrics, metrics_logger, onnx_models,
                      partitioner, telemetry (helpers used by FL runtime)
    configs/          configuration files for FL runs
    scripts/          scripts to run FL workers/coordinator
    requirements.txt  dependencies specific to FL part
  tabular/            feature extraction from binaries/ELF
  training/           local training scripts for lightgbm and siamese svm
models/
  lightgbm_model.pkl
  siamese_svm.pkl
  onnx/               ONNX artifacts produced by export
```

## Training Pipeline
1. Data Collection and Processing: A dataset of 500 malware and benign binaries is collected 
2. Tabular Feature Classification: The structural features, and metadata are extracted to create a tabular dataset. This data is used to train a LightGBM classifier 
3. Assembly Feature Classification: Raw assembly instructions are extracted, tokenized to N-grams and converted to TF-IDF matrix. This matrix is used to train the Siamese SVM classifier 
4. Model Training and Lightweight inference: The classifiers are trained using the tabular dataset and TF-IDF matrix. This is converted to a lightweight edge inference format using ONNX. 

Setup 
--------------
1. Install dependancies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r src/fl/requirements.txt   # optional: FL-specific deps
```

2. Extract tabular features
```bash
python src/tabular/elf_feature_extractor.py  <path to elf directory> --output <path to output csv>
```

3. Train a LightGBM model locally
```bash
python src/training/train_lightgbm.py --csv_path <path to csv> 
```

4. Train Siamese+SVM pipeline
```bash
python src/training/train_siamese_svm.py 
```

5. Export model to ONNX (for deployment)
```bash
python src/export/export_models_onnx.py --input models/lightgbm_model.pkl --output models/onnx/lightgbm.onnx
```

6. Container build (optional)
```bash
docker build -t iot-security-fl:latest .
# docker run ... (mount data/models as needed)
```

