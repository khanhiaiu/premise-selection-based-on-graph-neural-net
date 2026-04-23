import os
from huggingface_hub import snapshot_download

def download_retrieval_model():
    repo_id = "ruc-ai4math/Lean_State_Search_Random"
    # The folder we identified in research
    subfolder = "Finetune_Model/410_stable_random_s0_d0"
    local_dir = "models/flag_model"
    
    print(f"Downloading {subfolder} from {repo_id}...")
    print(f"Target directory: {local_dir}")
    
    os.makedirs(local_dir, exist_ok=True)
    
    # snapshot_download allows downloading a specific subfolder
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subfolder}/*"],
        local_dir="tmp_download",
    )
    
    # Move files to the final directory for cleaner structure
    import shutil
    src_path = os.path.join("tmp_download", subfolder)
    for item in os.listdir(src_path):
        s = os.path.join(src_path, item)
        d = os.path.join(local_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
            
    # Clean up tmp
    shutil.rmtree("tmp_download")
    
    print("\nDownload complete!")
    print(f"Model files are located in: {os.path.abspath(local_dir)}")

if __name__ == "__main__":
    download_retrieval_model()
