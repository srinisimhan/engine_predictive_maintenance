# Engine Predictive Maintenance - MLOps Pipeline

## Project Overview

This project implements a complete MLOps pipeline for predicting engine health conditions (**Normal vs Faulty**) using sensor telemetry data. The system automates data registration, preprocessing, feature engineering, model training, experiment tracking, bulk testing, and deployment using Hugging Face Spaces with CI/CD via GitHub Actions.


## Business Problem

Unexpected engine failures can cause unplanned downtime, high maintenance costs, and safety risks. Traditional reactive maintenance approaches are inefficient and often lead to late fault detection.
This solution provides an **automated, data-driven predictive maintenance system** that identifies faulty engine conditions early, enabling proactive maintenance and improved operational reliability.


## Dataset

The dataset contains engine operational records collected from multiple sensors, including:

* **Mechanical indicators**: Engine RPM, Fuel Pressure, Lubrication Oil Pressure
* **Thermal indicators**: Lubrication Oil Temperature, Coolant Temperature
* **Cooling indicators**: Coolant Pressure
* **Engineered features**:
  * RPM–Fuel Pressure Ratio
  * Power Index
  * Thermal Pressure Index
  * Mechanical Cooling Balance
  * Pressure Coordination
  * Operational condition flags
* **Target variable**: `Engine Condition`
  * `0` = Normal
  * `1` = Faulty


## Project Architecture
```
engine_predictive_maintenance/
├── .github/workflows/
│ └── pipeline.yml                          # CI/CD Pipeline Configuration
├── predictive_maintenance/                 # Main Project Directory
│ ├── data/                                 # Raw and processed datasets
│ │ ├── engine_data.csv                     # Original dataset
│ │ ├── bulk_test_sample.csv                # Bulk Test sample dataset
│ │ ├── train_prepared_raw.csv              # Training data (raw features)
│ │ ├── test_prepared_raw.csv               # Testing data (raw features)
│ │ ├── train_prepared_scaled.csv           # Training data (scaled)
│ │ └── test_prepared_scaled.csv            # Testing data (scaled)
│ ├── model_building/                       # ML Pipeline Scripts
│ │ ├── data_register.py                    # Dataset registration to HF
│ │ ├── prep.py                             # Data preprocessing & feature engineering
│ │ └── train.py                            # Model training with MLflow tracking
│ ├── deployment/                           # Deployment Files
│ │ ├── app.py                              # Streamlit application
│ │ ├── bulk_data_upload.py                 # Bulk prediction utility
│ │ ├── Dockerfile                          # Container configuration
│ │ └── requirements.txt                    # Deployment dependencies
│ ├── hosting/                              # Hosting Scripts
│ │ └── hosting.py                          # Deployment to HF Spaces
│ └── requirements.txt                      # GitHub Actions dependencies
└── README.md                               # Project documentation
```

## Live Deployments

### **Hugging Face Spaces**
- **Application**:
  [Engine_Predictive_Maintenance_App](https://huggingface.co/spaces/simnid/Engine-Predictive-Maintenance)
- **Model**:
  [predictive-maintenance-model](https://huggingface.co/simnid/predictive-maintenance-model)
- **Dataset**:
  [predictive-engine-maintenance-dataset](https://huggingface.co/datasets/simnid/predictive-engine-maintenance-dataset)
- **Bulk CSV Sample**: [bulk_test_sample.csv](https://huggingface.co/datasets/simnid/predictive-engine-maintenance-dataset/blob/main/bulk_test_sample.csv)
---

## Technical Implementation

### **1. Data Pipeline**
* **Data Registration**: Automated dataset upload to Hugging Face Hub
* **Data Preparation**:
  * Missing value handling
  * Feature scaling using `StandardScaler`
  * Domain-driven feature engineering
  * Train–test split with reproducibility
* **Data Storage**: Versioned datasets hosted on Hugging Face

### **2. Machine Learning Model**

* **Algorithm**: Gradient Boosting Classifier
* **Feature Engineering**:
  * Ratio-based mechanical features
  * Thermal and pressure interaction indices
  * Binary condition flags
* **Hyperparameter Tuning**: GridSearchCV
* **Primary Metric**:
  * Recall-Faulty
* **Reported Metrics:**
  * ROC-AU, PR-AUC, F1-Score
  * Decision Threshold


### **3. MLOps Components**
* **Experiment Tracking**: MLflow for logging parameters and metrics
* **Model Registry**: Hugging Face Model Hub
* **Containerization**: Docker-based deployment
* **CI/CD**: GitHub Actions with automated multi-stage pipeline

### **4. Deployment Stack**
* **Frontend**: Streamlit web application
* **Backend**: Gradient Boosting model with engineered feature pipeline
* **Hosting**: Hugging Face Spaces 
* **Inference Capabilities**:

  * Single engine prediction
  * Bulk CSV upload for batch predictions
  * Probability-based fault interpretation


## Model Performance

| Metric                 | Value |
| ---------------------- | ----- |
| **Recall**             | ~0.84 |
| **ROC-AUC**            | ~0.70 |
| **PR-AUC**             | ~0.80 |

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/pipeline.yml`) automates:

1. **Register Dataset**: Upload raw engine data to Hugging Face
2. **Data Preparation**: Clean, engineer features, and split data
3. **Bulk CSV Creation & Upload**: Generate sample bulk prediction files
4. **Model Training**: Train model with hyperparameter tuning and MLflow tracking
5. **Model Registration**: Upload best model to Hugging Face Model Hub
6. **Deploy to HF Space**: Deploy Streamlit application to Hugging Face

### **Secrets Required**

* `HF_TOKEN`: Hugging Face authentication token (write access)

## Local Development

### **Prerequisites**
* Python 3.10+
* Git
* Hugging Face account with write token

### **Setup**
```bash
# Clone repository
git clone https://github.com/srinisimhan/engine_predictive_maintenance.git
cd engine_predictive_maintenance

# Install dependencies
pip install -r predictive_maintenance/requirements.txt

# Set environment variable
export HF_TOKEN="your_huggingface_token"
```

## Using the Application

* **Single Prediction**:
  Enter engine sensor values and click **Predict Engine Condition**
* **Prediction Output**:
  * Normal / Faulty classification
  * Probability of Faulty
* **Bulk CSV Prediction**:
  * Upload CSV file with sensor readings
  * Download prediction results directly from the app

### Sample Prediction Scenarios
* **Normal Engine**: Stable RPM, moderate temperatures, balanced pressures
* **Faulty Engine**: Low oil pressure, high coolant temperature, abnormal RPM patterns

## Monitoring & Maintenance
### MLflow Tracking
* Track experiment history
* Compare hyperparameters and metrics
* Maintain reproducibility across training runs

### Model Retraining
The CI/CD pipeline automatically retrains the model when:
* New data is added
* Code updates are pushed to the `main` branch

## Troubleshooting
### Common Issues
1. **HF_TOKEN errors**: Ensure token has write permissions
2. **Model loading failures**: Verify model repo name and artifact path
3. **Dependency issues**: Confirm versions in `requirements.txt`
4. **Streamlit startup delays**: Allow Docker build to complete

### Debug Steps
```bash
# Test Hugging Face connectivity
python -c "from huggingface_hub import HfApi; print('Connected to HF')"

# Test dependency availability
python -c "import xgboost, joblib; print('Dependencies OK')"

# Run Streamlit locally
streamlit run predictive_maintenance/deployment/app.py
```

## References
* Hugging Face Hub Documentation
  [https://huggingface.co/docs/hub/index](https://huggingface.co/docs/hub/index)
* MLflow Documentation
  [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)
* Streamlit Documentation
  [https://docs.streamlit.io/](https://docs.streamlit.io/)


