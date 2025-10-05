#!/usr/bin/env python3
"""
Continuously sync outputs folder to Hugging Face Hub.

This script uploads the entire outputs folder and then monitors for new files,
uploading them every 5 minutes.

Usage:
    python sync_outputs_to_hf.py --repo-id your-username/repo-name
"""

import os
import time
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, upload_folder, upload_file


class OutputsSyncer:
    def __init__(self, outputs_path, repo_id, token=None, private=False):
        self.outputs_path = Path(outputs_path)
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.private = private
        self.state_file = self.outputs_path.parent / ".hf_sync_state.json"
        self.uploaded_files = self.load_state()
        
        # Create repository if it doesn't exist
        try:
            self.api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            print(f"Repository '{repo_id}' is ready.")
        except Exception as e:
            print(f"Note: {e}")
    
    def load_state(self):
        """Load the state of previously uploaded files."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_state(self):
        """Save the state of uploaded files."""
        with open(self.state_file, 'w') as f:
            json.dump(self.uploaded_files, f, indent=2)
    
    def get_file_hash(self, filepath):
        """Get MD5 hash of a file for change detection."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_all_files(self):
        """Get all files in the outputs directory."""
        files = {}
        for filepath in self.outputs_path.rglob("*"):
            if filepath.is_file() and not filepath.name.startswith('.'):
                rel_path = filepath.relative_to(self.outputs_path)
                try:
                    files[str(rel_path)] = {
                        'path': str(filepath),
                        'hash': self.get_file_hash(filepath),
                        'mtime': os.path.getmtime(filepath)
                    }
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
        return files
    
    def initial_upload(self):
        """Perform initial upload of the entire outputs folder."""
        print(f"\n📤 Starting initial upload of {self.outputs_path} to {self.repo_id}...")
        
        try:
            upload_folder(
                folder_path=str(self.outputs_path),
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Initial upload of outputs folder - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Record all files as uploaded
            self.uploaded_files = self.get_all_files()
            self.save_state()
            
            print(f"✅ Initial upload completed successfully!")
            print(f"   Uploaded {len(self.uploaded_files)} files")
            
        except Exception as e:
            print(f"❌ Error during initial upload: {e}")
            raise
    
    def sync_new_files(self):
        """Check for new or modified files and upload them."""
        current_files = self.get_all_files()
        new_or_modified = []
        
        # Find new or modified files
        for rel_path, file_info in current_files.items():
            if rel_path not in self.uploaded_files:
                new_or_modified.append((rel_path, file_info, "new"))
            elif file_info['hash'] != self.uploaded_files[rel_path]['hash']:
                new_or_modified.append((rel_path, file_info, "modified"))
        
        if not new_or_modified:
            return 0
        
        print(f"\n🔄 Found {len(new_or_modified)} new/modified files to upload...")
        
        # Upload each new or modified file
        for rel_path, file_info, status in new_or_modified:
            try:
                print(f"  📤 Uploading {status} file: {rel_path}")
                
                upload_file(
                    path_or_fileobj=file_info['path'],
                    path_in_repo=f"outputs/{rel_path}",
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"Update {rel_path} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                # Update state
                self.uploaded_files[rel_path] = file_info
                
            except Exception as e:
                print(f"  ❌ Error uploading {rel_path}: {e}")
        
        # Save updated state
        self.save_state()
        print(f"✅ Sync completed! Uploaded {len(new_or_modified)} files")
        
        return len(new_or_modified)
    
    def run_continuous_sync(self, interval_minutes=5):
        """Run continuous synchronization."""
        print(f"\n🚀 Starting continuous sync mode (checking every {interval_minutes} minutes)")
        print(f"   Monitoring: {self.outputs_path}")
        print(f"   Repository: {self.repo_id}")
        print("\n   Press Ctrl+C to stop\n")
        
        # Perform initial upload if this is the first run
        if not self.uploaded_files:
            self.initial_upload()
        else:
            print(f"📊 Resuming from previous state ({len(self.uploaded_files)} files already uploaded)")
            # Do an immediate sync check
            self.sync_new_files()
        
        # Continuous monitoring loop
        try:
            while True:
                print(f"\n⏰ Waiting {interval_minutes} minutes until next check... ({datetime.now().strftime('%H:%M:%S')})")
                time.sleep(interval_minutes * 60)
                
                print(f"\n🔍 Checking for new files... ({datetime.now().strftime('%H:%M:%S')})")
                uploaded_count = self.sync_new_files()
                
                if uploaded_count == 0:
                    print("   No new files found.")
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Sync stopped by user.")
            print(f"   Total files tracked: {len(self.uploaded_files)}")
            print(f"   State saved to: {self.state_file}")


def main():
    parser = argparse.ArgumentParser(description="Continuously sync outputs folder to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True,
                      help="Hugging Face repository ID (e.g., username/repo-name)")
    parser.add_argument("--outputs-path", type=str, default="baselines/il_lib/outputs",
                      help="Path to outputs folder (default: baselines/il_lib/outputs)")
    parser.add_argument("--interval", type=int, default=5,
                      help="Check interval in minutes (default: 5)")
    parser.add_argument("--token", type=str, default=None,
                      help="Hugging Face API token (or set HF_TOKEN env variable)")
    parser.add_argument("--private", action="store_true",
                      help="Make the repository private")
    parser.add_argument("--once", action="store_true",
                      help="Run once instead of continuous monitoring")
    
    args = parser.parse_args()
    
    # Validate outputs path
    if not os.path.exists(args.outputs_path):
        raise ValueError(f"Outputs path not found: {args.outputs_path}")
    
    # Get token from argument or environment
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: No HF token provided. You may need to login with 'huggingface-cli login'")
    
    # Create syncer
    syncer = OutputsSyncer(
        outputs_path=args.outputs_path,
        repo_id=args.repo_id,
        token=token,
        private=args.private
    )
    
    # Run sync
    if args.once:
        # One-time sync
        if not syncer.uploaded_files:
            syncer.initial_upload()
        else:
            syncer.sync_new_files()
    else:
        # Continuous sync
        syncer.run_continuous_sync(interval_minutes=args.interval)


if __name__ == "__main__":
    main()
