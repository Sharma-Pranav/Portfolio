from huggingface_hub import HfApi

repo_name = "dynamic-pricing-model"
api = HfApi()

# Create a new repository
api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=False)
