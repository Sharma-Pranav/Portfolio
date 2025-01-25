from huggingface_hub import HfApi, create_repo, upload_folder
import os

# Ensure HUGGINGFACE_TOKEN is set in the environment
token = os.getenv("HUGGINGFACE_TOKEN")

# Create or access the Hugging Face repository
api = HfApi()
repo_url = create_repo(
    name="RidePricingInsightEngine",
    token=token,
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
)

# Define the folder containing your project files
source_folder = "projects/DynamicPricingOptimization/"

# Upload the folder to the Hugging Face Space
upload_folder(
    folder_path=source_folder,
    repo_id="PranavSharma/RidePricingInsightEngine",
    repo_type="space",
    token=token,
)