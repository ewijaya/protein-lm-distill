from huggingface_hub import HfApi

# Abbreviates the dataset names (cluspro, propedia, pepnn) as "cppp

api = HfApi()
repo_id = "littleworth/protgpt2-distilled-medium"
api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

model_dir = "/home/ubuntu/storage1/distilling_protgpt2/models/protgpt2-distilled-t10.0-a0.1-l12-h16-e1024-p0.1-lr1e-04.uniprot"
api.upload_folder(
    repo_id=repo_id, repo_type="model", folder_path=model_dir, revision="main"
)
