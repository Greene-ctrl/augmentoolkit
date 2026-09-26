# Augmentoolkit: Resumable Private Hugging Face Space Dataset Factory

This document describes the architecture, lifecycle, configuration, and operational semantics for deploying Augmentoolkit as a resilient, private Hugging Face Space dataset factory.

---

## 1. Architectural Overview & Privacy Assumptions

The Hugging Face Space filesystem is treated as **ephemeral** and non-authoritative. A Space restart, rebuild, or sleep event can wipe local disk state at any time without warning.

To guarantee zero data loss during long-running dataset generation jobs:
1. **HF_SOURCE_REPO**: A private Hugging Face dataset repository containing raw or preprocessed source input documents.
2. **HF_WORK_REPO**: A private Hugging Face dataset repository storing durable job manifests, intermediate stage/shard checkpoints, and job execution logs.
3. **HF_OUTPUT_REPO**: A private Hugging Face dataset repository where final generated training datasets (e.g. Parquet/JSONL) are written.

### Security & Credentials
- All three repositories **must be private**.
- `HF_TOKEN` is supplied **exclusively** via Hugging Face Space Secrets or environment variables.
- Credentials are **never** hard-coded into code or committed to repository branches.

---

## 2. Environment Variables & Space Secrets

Configure the following secrets/variables in your Hugging Face Space settings:

| Variable | Type | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | Secret | Hugging Face Access Token with read/write access to dataset repos. |
| `HF_SOURCE_REPO` | Secret/Env | Private dataset repository for input documents (e.g. `your-org/private-source`). |
| `HF_WORK_REPO` | Secret/Env | Private dataset repository for manifests & checkpoints (e.g. `your-org/private-work`). |
| `HF_OUTPUT_REPO` | Secret/Env | Private dataset repository for final datasets (e.g. `your-org/private-output`). |
| `HF_AUTO_RESUME` | Env | Set to `true` to enable automatic resume of incomplete jobs upon Space startup. |
| `REDIS_HOST` | Env | Hostname for Redis queue server (default: `localhost`). |
| `REDIS_PORT` | Env | Port for Redis queue server (default: `6379`). |

---

## 3. Job Manifest & Durable State

For every pipeline run, a durable job manifest JSON is maintained in `HF_WORK_REPO` under `jobs/{job_id}/manifest.json`.

The manifest records:
- `job_id`: Unique identifier for the generation job.
- `pipeline`: Pipeline alias/node path (e.g., `factual-datagen-pipeline`).
- `source_repo`, `source_revision`, `source_path`, `source_split`: Source document reference.
- `work_repo`: Reference to intermediate work repository.
- `output_repo`: Reference to final output dataset repository.
- `configuration_hash`: SHA256 hash of merged/flattened configuration parameters.
- `prompt_config_hash`: SHA256 hash of pipeline prompt templates.
- `code_revision`: Git commit SHA or version string of Augmentoolkit.
- `current_stage` & `current_shard`: Current stage and work unit index.
- `completed_shards`: List of confirmed completed work units.
- `status`: Job status (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `REVOKED`).
- `timestamps`: Timestamps for creation, last update, and completion.
- `final_output_revision`: Commit SHA of final dataset in `HF_OUTPUT_REPO`.

---

## 4. Remote-First Checkpointing & Fail-Closed Semantics

Checkpoint persistence is **remote-first**:
1. Intermediate step outputs and shard results are saved locally in temporary JSON files.
2. The checkpoint file and updated `manifest.json` are uploaded immediately to `HF_WORK_REPO`.
3. **Verification**: The system verifies remote existence on Hugging Face Hub before marking the checkpoint durable in the manifest.
4. **Local Cleanup**: Temporary local files are deleted **only after** successful remote confirmation.
5. **Fail-Closed Behavior**: If remote checkpoint upload or verification fails, the job fails closed (marks state `FAILED` and raises an exception) rather than silently continuing or incorrectly marking work complete.

---

## 5. Stage/Shard Resumability & Recovery

Jobs are partitioned into deterministic work units/shards:
- **Completed Shards**: Listed in `completed_shards` in `manifest.json`. After a Space restart, completed shards are **never regenerated**.
- **Partially Completed Shards**: Resumed from the latest usable checkpoint if supported; otherwise, only that specific work unit is restarted (not the entire dataset).
- **Startup Reconciliation**:
  1. On startup, the API connects to `HF_WORK_REPO` and discovers incomplete jobs (`PENDING`, `RUNNING`, `INTERRUPTED`).
  2. The system checks compatibility between the remote manifest and target run parameters.
  3. **Incompatible Resumes**: Continuation is refused if `source_revision`, `configuration_hash`, `prompt_config_hash`, or `code_revision` materially changed. A new or forked run is required instead of mixing incompatible generations.

---

## 6. Launching and Managing Jobs via API

### Launch a New Pipeline Job
```bash
curl -X POST "https://<your-space>.hf.space/pipelines/run" \
     -H "Content-Type: application/json" \
     -d '{
           "node_path": "factual-datagen-pipeline",
           "config_path": "external:_START_HERE_complete_factual.yaml",
           "parameters": {
             "hf_source_repo": "your-org/private-source",
             "hf_work_repo": "your-org/private-work",
             "hf_output_repo": "your-org/private-output",
             "source_path": "raw_documents",
             "source_revision": "main"
           }
         }'
```

### Readiness & Health Reporting
Check system readiness, including Redis, Huey worker, HF authentication, and repository accessibility:
```bash
curl "https://<your-space>.hf.space/readiness"
```

### Inspect Jobs & Manifests
```bash
# List all jobs in work repo
curl "https://<your-space>.hf.space/jobs?work_repo=your-org/private-work"

# Inspect job manifest
curl "https://<your-space>.hf.space/jobs/<job_id>/manifest"

# Inspect checkpoint references
curl "https://<your-space>.hf.space/jobs/<job_id>/checkpoints"

# Inspect final output references
curl "https://<your-space>.hf.space/jobs/<job_id>/outputs"
```

### Explicitly Resume an Incomplete Job
```bash
curl -X POST "https://<your-space>.hf.space/jobs/<job_id>/resume" \
     -H "Content-Type: application/json" \
     -d '{}'
```
