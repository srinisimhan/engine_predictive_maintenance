# Import Packages
from huggingface_hub import HfApi
import os

# HF Authentication
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()
else:
    raise EnvironmentError("HF_TOKEN not set!")

api = HfApi(token=HF_TOKEN)

# Space must already exist (created manually in Hugging Face UI)
SPACE_REPO_ID = "simnid/Engine-Predictive-Maintenance"
REPO_TYPE = "space"

# Validate Space exists before upload
api.repo_info(repo_id=SPACE_REPO_ID, repo_type=REPO_TYPE)
print(f"Found Space: {SPACE_REPO_ID}")

# Upload deployment folder contents to the Space root
api.upload_folder(
    folder_path="predictive_maintenance/deployment",
    repo_id=SPACE_REPO_ID,
    repo_type=REPO_TYPE,
    path_in_repo=""
)

print("Deployment files uploaded successfully.")
print(f"Space URL: https://huggingface.co/spaces/{SPACE_REPO_ID}")
