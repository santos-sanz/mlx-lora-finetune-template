"""
Hugging Face Hub utilities for model download and upload.
"""

import os
from pathlib import Path
from typing import Optional, Union, List
from huggingface_hub import HfApi, snapshot_download, upload_folder, create_repo
from dotenv import load_dotenv


def get_hf_token() -> Optional[str]:
    """Get Hugging Face token from environment."""
    load_dotenv()
    return os.getenv("HF_TOKEN")


def check_repo_exists(repo_id: str, token: Optional[str] = None) -> dict:
    """
    Check if a HuggingFace repository exists and get info.
    
    Args:
        repo_id: Repository ID to check
        token: HF token for private repos
    
    Returns:
        Dictionary with 'exists' boolean and repo info if exists
    """
    from huggingface_hub.utils import RepositoryNotFoundError
    
    token = token or get_hf_token()
    api = HfApi(token=token)
    
    try:
        info = api.repo_info(repo_id=repo_id, token=token)
        return {
            "exists": True,
            "private": info.private,
            "last_modified": str(info.last_modified) if info.last_modified else None,
            "sha": info.sha,
            "downloads": getattr(info, 'downloads', 0),
            "likes": getattr(info, 'likes', 0),
            "siblings": len(info.siblings) if info.siblings else 0,  # number of files
        }
    except RepositoryNotFoundError:
        return {"exists": False}
    except Exception as e:
        return {"exists": False, "error": str(e)}


def download_model(
    model_id: str,
    local_dir: Optional[Union[str, Path]] = None,
    token: Optional[str] = None,
    revision: str = "main",
) -> Path:
    """
    Download model from Hugging Face Hub.
    
    Args:
        model_id: Hugging Face model identifier (e.g., 'meta-llama/Llama-3.2-1B')
        local_dir: Local directory to save model (default: cache)
        token: HF token for private models
        revision: Git revision to download
    
    Returns:
        Path to downloaded model directory
    """
    token = token or get_hf_token()
    
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        token=token,
        revision=revision,
    )
    
    print(f"Downloaded model '{model_id}' to {path}")
    return Path(path)


def upload_model(
    model_path: Union[str, Path],
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: str = "Upload fine-tuned model",
) -> str:
    """
    Upload trained model to Hugging Face Hub.
    
    Args:
        model_path: Path to model directory
        repo_id: Target repository ID (e.g., 'username/model-name')
        token: HF token for authentication
        private: Whether to make the repository private
        commit_message: Commit message for the upload
    
    Returns:
        URL of the uploaded model
    """
    token = token or get_hf_token()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload folder
    upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        token=token,
        commit_message=commit_message,
    )
    
    print(f"Uploaded model to https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


def upload_checkpoint(
    checkpoint_path: Union[str, Path],
    repo_id: str,
    checkpoint_name: Optional[str] = None,
    token: Optional[str] = None,
    private: bool = True,
) -> str:
    """
    Upload a specific checkpoint to Hugging Face Hub.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        repo_id: Target repository ID
        checkpoint_name: Name for the checkpoint folder in repo
        token: HF token for authentication
        private: Whether to make the repository private
    
    Returns:
        URL of the uploaded checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_name = checkpoint_name or checkpoint_path.name
    
    token = token or get_hf_token()
    
    # Create repo if needed
    try:
        create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
    except Exception:
        pass
    
    # Upload to subfolder
    url = upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        path_in_repo=f"checkpoints/{checkpoint_name}",
        token=token,
        commit_message=f"Upload checkpoint: {checkpoint_name}",
    )
    
    print(f"Uploaded checkpoint to https://huggingface.co/{repo_id}/tree/main/checkpoints/{checkpoint_name}")
    return url


def list_checkpoints(repo_id: str, token: Optional[str] = None) -> List[str]:
    """
    List available checkpoints in a Hugging Face repository.
    
    Args:
        repo_id: Repository ID to list checkpoints from
        token: HF token for private repos
    
    Returns:
        List of checkpoint names
    """
    token = token or get_hf_token()
    api = HfApi(token=token)
    
    try:
        files = api.list_repo_files(repo_id=repo_id, token=token)
        checkpoints = set()
        for f in files:
            if f.startswith("checkpoints/"):
                parts = f.split("/")
                if len(parts) > 1:
                    checkpoints.add(parts[1])
        return sorted(list(checkpoints))
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return []
