
"""
Production Model Training Script
--------------------------------
This script performs end-to-end model training with:
- Data loading from Hugging Face
- Comprehensive hyperparameter tuning
- MLflow experiment tracking
- Model evaluation and artifact generation
- Model registration to Hugging Face Model Hub
"""

# Imports
# for creating a folder
import os


# for data manipulation
import pandas as pd
import numpy as np

# ML libraries
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    recall_score,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

# Model serialization
import joblib

# Experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Hugging Face integration
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError


# Configuration
DATA_REPO_ID = "simnid/predictive-engine-maintenance-dataset"
MODEL_REPO_ID = "simnid/predictive-maintenance-model"

TARGET_COL = "Engine Condition"
RANDOM_STATE = 42

LOCAL_DATA_DIR = "predictive_maintenance/data"
ARTIFACT_DIR = "predictive_maintenance/artifacts"

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")

# Strip any potential whitespace from the token before check
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable not set!")

# MLflow setup
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "predictive-maintenance-production"

try:
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    print("MLflow experiment setup failed, falling back to default experiment.")
    print(str(e))


# Data Loading from Hugging Face
def download_and_load_csv(filename):
    local_path = os.path.join(LOCAL_DATA_DIR, filename)

    if not os.path.exists(local_path):
        print(f"Downloading {filename} from Hugging Face dataset repo...")
        hf_hub_download(
            repo_id=DATA_REPO_ID,
            filename=filename,
            local_dir=LOCAL_DATA_DIR,
            repo_type="dataset",
            token=HF_TOKEN
        )

    return pd.read_csv(local_path)


def load_prepared_data():
    print("Loading prepared datasets from Hugging Face")

    train_df = download_and_load_csv("train_prepared_raw.csv")
    test_df = download_and_load_csv("test_prepared_raw.csv")

    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape:  {test_df.shape}")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].astype(int)

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].astype(int)

    return X_train, X_test, y_train, y_test


# Model Training & Evaluation
def train_and_evaluate_models():

    X_train, X_test, y_train, y_test = load_prepared_data()

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance scale_pos_weight: {scale_pos_weight:.2f}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    best_models = {}
    model_scores = {}

    # Random Forest
    print("\nTraining Random Forest")
    with mlflow.start_run(run_name="RandomForest_Production"):

        rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        param_grid_rf = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

        rf_gs = GridSearchCV(
            rf,
            param_grid_rf,
            cv=cv,
            scoring="recall",
            n_jobs=-1
        )

        rf_gs.fit(X_train, y_train)
        best_rf = rf_gs.best_estimator_

        for p, v in rf_gs.best_params_.items():
            mlflow.log_param(f"rf_{p}", v)

        y_pred = best_rf.predict(X_test)
        y_proba = best_rf.predict_proba(X_test)[:, 1]

        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        precision, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision)

        mlflow.log_metrics({
            "recall_faulty": recall,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })
        input_example = X_train.iloc[:5]
        mlflow.sklearn.log_model(best_rf, name="random_forest_model", input_example=input_example)

        best_models["RandomForest"] = best_rf
        model_scores["RandomForest"] = recall

    # XGBoost
    print("\nTraining XGBoost")
    with mlflow.start_run(run_name="XGBoost_Production"):

        xgb_clf = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1
        )

        param_grid_xgb = {
            "n_estimators": [100, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

        xgb_gs = GridSearchCV(
            xgb_clf,
            param_grid_xgb,
            cv=cv,
            scoring="recall",
            n_jobs=-1
        )

        xgb_gs.fit(X_train, y_train)
        best_xgb = xgb_gs.best_estimator_

        for p, v in xgb_gs.best_params_.items():
            mlflow.log_param(f"xgb_{p}", v)

        y_pred = best_xgb.predict(X_test)
        y_proba = best_xgb.predict_proba(X_test)[:, 1]

        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        precision, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision)

        mlflow.log_metrics({
            "recall_faulty": recall,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })
        input_example = X_train.iloc[:5]
        mlflow.xgboost.log_model(best_xgb, name="xgboost_model", input_example=input_example)

        best_models["XGBoost"] = best_xgb
        model_scores["XGBoost"] = recall

    # Gradient Boosting
    print("\nTraining Gradient Boosting")
    with mlflow.start_run(run_name="GradientBoosting_Production"):

        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

        param_grid_gb = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 4]
        }

        gb_gs = GridSearchCV(
            gb,
            param_grid_gb,
            cv=cv,
            scoring="recall",
            n_jobs=-1
        )

        gb_gs.fit(X_train, y_train)
        best_gb = gb_gs.best_estimator_

        for p, v in gb_gs.best_params_.items():
            mlflow.log_param(f"gb_{p}", v)

        y_pred = best_gb.predict(X_test)
        y_proba = best_gb.predict_proba(X_test)[:, 1]

        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        precision, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision)

        mlflow.log_metrics({
            "recall_faulty": recall,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })
        input_example = X_train.iloc[:5]

        mlflow.sklearn.log_model(best_gb, name="gradient_boosting_model", input_example=input_example)

        best_models["GradientBoosting"] = best_gb
        model_scores["GradientBoosting"] = recall

    return best_models, model_scores


# Hugging Face Model Registration
def register_best_model(best_models, model_scores):

    best_model_name = max(model_scores, key=model_scores.get)
    best_model = best_models[best_model_name]

    print(f"\nSelected Best Model (Recall-Optimized): {best_model_name}")

    model_path = os.path.join(
        ARTIFACT_DIR, "best_predictive_maintenance_model.joblib"
    )
    joblib.dump(best_model, model_path)

    api = HfApi(token=HF_TOKEN)

    try:
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_predictive_maintenance_model.joblib",
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )

    print(f"Model uploaded to Hugging Face: {MODEL_REPO_ID}")

    return best_model_name, model_path


# Main Execution
if __name__ == "__main__":
    print("Starting Production Model Training Pipeline")

    best_models, model_scores = train_and_evaluate_models()
    best_model_name, model_path = register_best_model(best_models, model_scores)

    print("\nTraining Complete")
    print(f"Best Model: {best_model_name}")
    print(f"Saved Model Path: {model_path}")
    print(f"Model Repo: https://huggingface.co/{MODEL_REPO_ID}")
    print(f"MLflow UI: {MLFLOW_TRACKING_URI}")
