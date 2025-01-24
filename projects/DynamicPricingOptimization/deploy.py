import os
from huggingface_hub import Repository

repo = Repository(
    local_dir='.',
    clone_from='PranavSharma/RidePricingInsightEngine',
    use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
)
repo.git_add(auto_lfs_track=True)
repo.git_commit(commit_message='Automated deployment')
repo.git_push()