from huggingface_hub import hf_hub_download
import os

model_repo = "Heralax/Augmentoolkit-DataSpecialist-v0.1-GGUF"
model_file = "Augmentoolkit-DataSpecialist-7.2B-Q8_0.gguf"
tokenizer_repo = "Heralax/Augmentoolkit-DataSpecialist-v0.1"

target_dir = "models/augmentoolkit-v0.1"
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
    hf_hub_download(
        repo_id=tokenizer_repo,
        filename=f,
        local_dir=target_dir
    )

print("Model and tokenizer files downloaded successfully.")
