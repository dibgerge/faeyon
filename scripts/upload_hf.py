import os
from pathlib import Path
from re import L
from huggingface_hub import HfApi, Repository, create_repo, upload_folder
from transformers import AutoModel, AutoTokenizer, AutoConfig


def upload_to_hub(
    repo_name: str = "dibgerges/faeyon",
    private: bool = False,
    commit_message: str = "Upload model and tokenizer",
    local_dir: str = None,
):
    """
    Upload a model to the Hugging Face Hub.
    
    Args:
        model_path (str): Path to the trained model directory
        repo_name (str): Name of the repository on the Hub
        organization (str, optional): Organization name if uploading to an org
        private (bool): Whether the repository should be private
        commit_message (str): Commit message for the upload
        local_dir (str, optional): Local directory to clone the repository to
    """   
    # Create repo (or get existing)
    repo_url = create_repo(
        repo_name,
        token=os.getenv("HF_TOKEN"),
        private=private,
        exist_ok=True,
    )
    
    print(f"Repository created at: {repo_url}")
    
    # Clone the repository locally if local_dir is provided
    # if local_dir:
    #     repo = Repository(
    #         local_dir=local_dir,
    #         clone_from=repo_url,
    #         use_auth_token=os.getenv("HF_TOKEN"),
    #     )
    
    # Load model and tokenizer
    # print("Loading model and tokenizer...")
    # model = AutoModel.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # config = AutoConfig.from_pretrained(model_path)
    
    # Save model and tokenizer to the repository
    save_path = local_dir if local_dir else "./"
    
    # print("Saving model and tokenizer...")
    # model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)
    
    # Create a README.md if it doesn't exist
    readme_path = os.path.join(save_path, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"---\n")
            f.write(f"language: en\n")
            f.write(f"license: apache-2.0\n")
            f.write(f"---\n\n")
            f.write(f"# {repo_name.split('/')[-1]}\n\n")
            f.write("## Model Description\n\n")
            f.write(f"This is a fine-tuned version of the VIT model.\n\n")
    
    # Upload to the Hub
    print("Uploading to the Hub...")
    # if local_dir:
    #     repo.push_to_hub(commit_message=commit_message)
    # else:
    upload_folder(
        path_in_repo=local_dir,
        repo_id=repo_name,
        folder_path=save_path,
        commit_message=commit_message,
        token=os.getenv("HF_TOKEN"),
    )
    
    print(f"Successfully uploaded to {repo_url}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload a model to the Hugging Face Hub")
    parser.add_argument("--repo_name", type=str, default="dibgerges/faeyon", help="Name of the repository on the Hub")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    parser.add_argument("--local_dir", type=str, default=None, help="Local directory to clone the repository to")
    parser.add_argument("--hf_token", type=str, help="Hugging Face authentication token")
    
    args = parser.parse_args()
    
    # Set the HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    upload_to_hub(
        repo_name=args.repo_name,
        private=args.private,
        local_dir=args.local_dir,
        
    )
