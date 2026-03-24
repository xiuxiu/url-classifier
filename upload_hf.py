"""
Upload url-classifier to HuggingFace Hub.
Run: python upload_hf.py
Requires: pip install huggingface_hub torch tiktoken
"""
import os
import sys

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable not set.")
    print("Run: set HF_TOKEN=hf_xxxx && python upload_hf.py")
    sys.exit(1)
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Optional: set proxy if needed
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

from huggingface_hub import HfApi, upload_file

REPO_ID = "xiuxiu/url-classifier-autoresearch"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(LOCAL_DIR, "url-autoresearch", "checkpoint_pre_eval.pt")
CONFIG_PATH = os.path.join(LOCAL_DIR, "configs", "model_config.json")

def main():
    api = HfApi()
    
    # Verify token
    user = api.whoami()
    print(f"Logged in as: {user['name']}")
    
    # Create repo
    print(f"Creating repo: {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    
    # Upload model card
    readme_path = os.path.join(LOCAL_DIR, "model_card.md")
    if os.path.exists(readme_path):
        print("Uploading model_card.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add model card",
        )
    
    # Upload config
    if os.path.exists(CONFIG_PATH):
        print("Uploading model_config.json...")
        api.upload_file(
            path_or_fileobj=CONFIG_PATH,
            path_in_repo="model_config.json",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add model config",
        )
    
    # Upload checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print("Uploading checkpoint (413MB)...")
        api.upload_file(
            path_or_fileobj=CHECKPOINT_PATH,
            path_in_repo="model.safetensors",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload trained weights",
        )
        print(f"\nDone! Model available at: https://huggingface.co/{REPO_ID}")
    else:
        print(f"Checkpoint not found at: {CHECKPOINT_PATH}")
        print("Please copy checkpoint_pre_eval.pt from url-autoresearch/ manually")

if __name__ == "__main__":
    main()
