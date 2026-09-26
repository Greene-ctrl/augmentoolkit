import os
import sys
from huggingface_hub import hf_hub_download

def main():
    # Only download model if explicitly requested via environment variable or CLI flag
    download_enabled = os.environ.get("DOWNLOAD_MODEL_ON_BUILD", "false").lower() in ("true", "1", "yes")
    if "--force" not in sys.argv and not download_enabled:
        print("Skipping default model download. Set DOWNLOAD_MODEL_ON_BUILD=true or pass --force to download.")
        return

    model_repo = os.environ.get("MODEL_REPO", "Heralax/Augmentoolkit-DataSpecialist-v0.1-GGUF")
    model_file = os.environ.get("MODEL_FILE", "Augmentoolkit-DataSpecialist-7.2B-Q8_0.gguf")
    tokenizer_repo = os.environ.get("TOKENIZER_REPO", "Heralax/Augmentoolkit-DataSpecialist-v0.1")

    target_dir = os.environ.get("MODEL_TARGET_DIR", "models/augmentoolkit-v0.1")
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading model {model_file} from {model_repo}...")
    hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        local_dir=target_dir
    )

    print(f"Downloading tokenizer files from {tokenizer_repo}...")
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json"
    ]

    for f in tokenizer_files:
        print(f"Downloading {f}...")
        try:
            hf_hub_download(
                repo_id=tokenizer_repo,
                filename=f,
                local_dir=target_dir
            )
        except Exception as e:
            print(f"Warning: Could not download {f}: {e}")

    print("Model and tokenizer process completed.")

if __name__ == "__main__":
    main()
