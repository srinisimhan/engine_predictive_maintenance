
"""
Data Preparation Script
-----------------------
This script prepares the engine sensor dataset for modeling by:
- Loading data from Hugging Face
- Applying EDA-justified feature engineering
- Creating both raw and scaled datasets
- Performing stratified train-test split
- Saving prepared datasets locally
- Uploading prepared datasets back to Hugging Face
"""

# Imports
# for file system operations
import os

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')  # This prevents figure display attempts

# for data manipulation
import pandas as pd
import numpy as np

# for preprocessing and splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# for Hugging Face integration
from huggingface_hub import HfApi
from datasets import load_dataset


# Configuration
REPO_ID = "simnid/predictive-engine-maintenance-dataset"
TARGET_COL = "Engine Condition"

RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_DIR = "predictive_maintenance/data"

TRAIN_RAW_FILE = "train_prepared_raw.csv"
TEST_RAW_FILE = "test_prepared_raw.csv"
TRAIN_SCALED_FILE = "train_prepared_scaled.csv"
TEST_SCALED_FILE = "test_prepared_scaled.csv"


# Load Dataset
print("Loading raw dataset from Hugging Face")
dataset = load_dataset(
    REPO_ID,
    data_files="engine_data.csv",
    split="train"
)

df = dataset.to_pandas()
print(f"Dataset shape: {df.shape}")


# Feature Engineering (EDA-Driven)
print("\nApplying feature engineering based on EDA findings")

df_eng = df.copy()

# Interaction features (Bivariate analysis)
df_eng["RPM_FuelPressure_Ratio"] = df["Engine rpm"] / (df["Fuel pressure"] + 1e-5)
df_eng["Power_Index"] = (df["Engine rpm"] * df["Fuel pressure"]) / 1000

# System stress indicators (PCA-informed trade-offs)
df_eng["Thermal_Pressure_Index"] = df["Coolant temp"] / (df["Fuel pressure"] + 1e-5)

df_eng["Mech_Cooling_Balance"] = (
    (df["Engine rpm"] + df["Lub oil pressure"]) -
    (df["Coolant temp"] + df["Coolant pressure"])
)

df_eng["Pressure_Coordination"] = (
    df["Fuel pressure"] - df["Coolant pressure"]
)

# Early warning flags (dataset-supported thresholds)
df_eng["Low_Oil_Pressure_Flag"] = (
    df["Lub oil pressure"] < 1.5
).astype(int)

df_eng["High_Coolant_Temp_Flag"] = (
    df["Coolant temp"] > 100
).astype(int)

df_eng["Low_RPM_Flag"] = (
    df["Engine rpm"] < 600
).astype(int)

print(f"Total features after engineering: {df_eng.shape[1]}")


# Train-Test Split
print("\nPerforming stratified train-test split")

features = [col for col in df_eng.columns if col != TARGET_COL]
X = df_eng[features]
y = df_eng[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Split validation
print("\nSplit Validation Report:")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Testing samples:  {X_test.shape[0]:,}")

print("\nClass Distribution:")
print(f"Training - Normal: {(y_train == 0).sum():,} ({100*(y_train == 0).mean():.1f}%)")
print(f"           Faulty:  {(y_train == 1).sum():,} ({100*(y_train == 1).mean():.1f}%)")
print(f"Testing  - Normal: {(y_test == 0).sum():,} ({100*(y_test == 0).mean():.1f}%)")
print(f"           Faulty:  {(y_test == 1).sum():,} ({100*(y_test == 1).mean():.1f}%)")


# Feature Scaling (for scale-sensitive models only)
print("\nApplying RobustScaler (train-fit only)")

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)


# Save Prepared Datasets
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Raw datasets (for tree-based models)
train_raw_df = pd.concat(
    [X_train.reset_index(drop=True),
     y_train.reset_index(drop=True)],
    axis=1
)

test_raw_df = pd.concat(
    [X_test.reset_index(drop=True),
     y_test.reset_index(drop=True)],
    axis=1
)

# Scaled datasets (for linear / distance-based models)
train_scaled_df = pd.concat(
    [X_train_scaled_df.reset_index(drop=True),
     y_train.reset_index(drop=True)],
    axis=1
)

test_scaled_df = pd.concat(
    [X_test_scaled_df.reset_index(drop=True),
     y_test.reset_index(drop=True)],
    axis=1
)

train_raw_path = os.path.join(OUTPUT_DIR, TRAIN_RAW_FILE)
test_raw_path = os.path.join(OUTPUT_DIR, TEST_RAW_FILE)
train_scaled_path = os.path.join(OUTPUT_DIR, TRAIN_SCALED_FILE)
test_scaled_path = os.path.join(OUTPUT_DIR, TEST_SCALED_FILE)

train_raw_df.to_csv(train_raw_path, index=False)
test_raw_df.to_csv(test_raw_path, index=False)
train_scaled_df.to_csv(train_scaled_path, index=False)
test_scaled_df.to_csv(test_scaled_path, index=False)

print("\nPrepared datasets saved locally:")
print(f"  {train_raw_path}")
print(f"  {test_raw_path}")
print(f"  {train_scaled_path}")
print(f"  {test_scaled_path}")


# Upload to Hugging Face
print("\nUploading prepared datasets to Hugging Face")

hf_token = os.getenv("HF_TOKEN")
if hf_token is not None:
    hf_token = hf_token.strip()

api = HfApi(token=hf_token)

api.upload_file(
    path_or_fileobj=train_raw_path,
    path_in_repo=TRAIN_RAW_FILE,
    repo_id=REPO_ID,
    repo_type="dataset"
)

api.upload_file(
    path_or_fileobj=test_raw_path,
    path_in_repo=TEST_RAW_FILE,
    repo_id=REPO_ID,
    repo_type="dataset"
)

api.upload_file(
    path_or_fileobj=train_scaled_path,
    path_in_repo=TRAIN_SCALED_FILE,
    repo_id=REPO_ID,
    repo_type="dataset"
)

api.upload_file(
    path_or_fileobj=test_scaled_path,
    path_in_repo=TEST_SCALED_FILE,
    repo_id=REPO_ID,
    repo_type="dataset"
)

print("\nData Preparation Complete")
print(f"Original dataset: {df.shape[0]:,} samples")
print(f"Prepared dataset: {train_raw_df.shape[0]:,} train + {test_raw_df.shape[0]:,} test")
print(f"Total features used for modeling: {len(features)}")
