#!/usr/bin/env python3
"""
Upload models and checkpoints to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py --model outputs/adapters/final
    python scripts/upload_to_hf.py --checkpoint outputs/checkpoints/step-1000
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.hf_utils import upload_model, upload_checkpoint, list_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="Upload to Hugging Face Hub")
    parser.add_argument("--model", type=str, help="Path to model directory to upload")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to upload")
    parser.add_argument("--repo-id", type=str, help="Target repo ID (overrides HF_REPO_ID)")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Make repository private (default: True)")
    parser.add_argument("--public", action="store_true",
                        help="Make repository public")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List checkpoints in the repository")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    
    # Get repo ID
    repo_id = args.repo_id or os.getenv("HF_REPO_ID")
    if not repo_id and not args.list_checkpoints:
        print("Error: No repository ID specified. Set HF_REPO_ID or use --repo-id")
        sys.exit(1)
    
    private = not args.public
    
    print("=" * 60)
    print("Hugging Face Hub Upload")
    print("=" * 60)
    
    if args.list_checkpoints:
        if not repo_id:
            print("Error: Need repo-id to list checkpoints")
            sys.exit(1)
        print(f"Listing checkpoints in {repo_id}...")
        checkpoints = list_checkpoints(repo_id)
        if checkpoints:
            print("Available checkpoints:")
            for cp in checkpoints:
                print(f"  - {cp}")
        else:
            print("No checkpoints found")
        return
    
    if args.model:
        print(f"Uploading model: {args.model}")
        print(f"Target: {repo_id}")
        print(f"Private: {private}")
        print("=" * 60)
        
        url = upload_model(
            model_path=args.model,
            repo_id=repo_id,
            private=private,
        )
        print("\nModel uploaded successfully!")
        print(f"URL: {url}")
    
    elif args.checkpoint:
        print(f"Uploading checkpoint: {args.checkpoint}")
        print(f"Target: {repo_id}")
        print("=" * 60)
        
        url = upload_checkpoint(
            checkpoint_path=args.checkpoint,
            repo_id=repo_id,
            private=private,
        )
        print("\nCheckpoint uploaded successfully!")
    
    else:
        print("Error: Specify --model or --checkpoint to upload")
        sys.exit(1)


if __name__ == "__main__":
    main()
