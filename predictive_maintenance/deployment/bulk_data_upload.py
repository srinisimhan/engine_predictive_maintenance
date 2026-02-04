# Importing packages
from huggingface_hub import HfApi
import os
import pandas as pd


# Create Bulk Test Sample Data
bulk_data = [
    # Engine rpm, Lub oil pressure, Fuel pressure, Coolant pressure, lub oil temp, Coolant temp
    [700, 2.49, 11.79, 3.18, 84.14, 81.63],
    [520, 2.96,  6.55, 1.06, 77.75, 79.65],
    [900, 3.50, 18.20, 2.90, 88.00, 95.00],
    [450, 1.20,  7.50, 2.00, 70.00, 110.0],  # high coolant temp + low oil pressure regime
    [1100, 4.10, 20.00, 3.50, 90.00, 85.00]
]

columns = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]

df_bulk = pd.DataFrame(bulk_data, columns=columns)

# Save locally inside data folder (consistent pattern)
local_path = "predictive_maintenance/data/bulk_test_sample.csv"
os.makedirs("predictive_maintenance/data", exist_ok=True)
df_bulk.to_csv(local_path, index=False)
print(f"Bulk CSV saved locally at {local_path}")

# Hugging Face Upload
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()
else:
    raise EnvironmentError("HF_TOKEN not set!")

DATA_REPO_ID = "simnid/predictive-engine-maintenance-dataset"
BULK_FILENAME = "bulk_test_sample.csv"

api = HfApi(token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=local_path,
    path_in_repo=BULK_FILENAME,
    repo_id=DATA_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN
)

print(f"Bulk CSV uploaded to Hugging Face dataset repo: {DATA_REPO_ID}/{BULK_FILENAME}")
