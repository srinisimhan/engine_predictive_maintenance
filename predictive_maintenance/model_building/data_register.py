# Importing relevant packages
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# HF Authentication
HF_TOKEN = os.getenv("HF_TOKEN")
# Strip any potential whitespace from the token
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()
api = HfApi(token=HF_TOKEN)

# Create a repository on hugging face to store the dataset
repo_id = "simnid/predictive-engine-maintenance-dataset"
repo_type = "dataset"

# Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Upload the folder to Hugging Face
api.upload_folder(
    folder_path="predictive_maintenance/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
