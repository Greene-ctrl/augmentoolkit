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

## 3. Durable Job ID vs. Ephemeral Task ID

Generation jobs maintain a strict separation between identifiers:
- **Durable `job_id`**: Identifies the dataset generation run across Space restarts, process terminations, and resume attempts.
- **Ephemeral `task_id`**: Assigned by the Huey task queue for a specific execution attempt.

When a job is resumed via `/jobs/{job_id}/resume` or auto-resume, the execution continues the existing manifest under the same durable `job_id`, appending the new Huey `task_id` to `task_history`. Active execution locks in Redis prevent duplicate concurrent execution of the same durable job.

---

## 4. Job Manifest & Durable State

For every pipeline run, a durable job manifest JSON is maintained in `HF_WORK_REPO` under `jobs/{job_id}/manifest.json`.

The manifest records:
- `job_id` & `current_task_id`: Durable job identifier and current execution task ID.
- `pipeline`: Pipeline alias/node path (e.g., `factual-datagen-pipeline`).
- `source_repo`, `source_revision`, `source_path`, `source_split`: Source document reference.
- `work_repo` & `output_repo`: References to intermediate work and output repositories.
- `configuration_hash`: SHA256 hash of merged/flattened configuration parameters.
- `prompt_config_hash`: SHA256 hash of pipeline prompt templates.
- `output_schema_version`: Version string for target output format.
- `code_revision`: Git commit SHA or version string of Augmentoolkit.
- `current_stage` & `current_shard`: Current stage and work unit index.
- `completed_shards`: List of confirmed completed work units.
- `checkpoints_metadata`: SHA256 checksums, byte sizes, and record counts for all completed shards.
- `output_shards`: Metadata for incrementally uploaded output shards.
- `status`: Job status (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `REVOKED`).
- `timestamps`: Timestamps for creation, last update, and completion.
- `final_output_revision`: Commit SHA of final dataset in `HF_OUTPUT_REPO`.

---

## 5. Remote-First Checkpointing & Fail-Closed Semantics

Checkpoint persistence is **remote-first**:
1. Intermediate step outputs and shard results are saved locally in temporary JSON files.
2. The checkpoint file (with computed SHA256 checksum) and updated `manifest.json` are uploaded immediately to `HF_WORK_REPO`.
3. **Verification**: The system verifies remote existence on Hugging Face Hub before marking the checkpoint durable in the manifest.
4. **Local Cleanup**: Temporary local files are deleted **only after** successful remote confirmation.
5. **Fail-Closed Semantics**: If `HF_WORK_REPO` is configured and initial manifest creation, checkpoint upload, or remote verification fails, the job fails closed immediately. It does **not** downgrade errors to warnings or start subprocesses without confirmed persistence.

### Corruption Safeguards
Upon recovery, downloaded remote checkpoints are verified against recorded SHA256 checksums and JSON integrity. Any corrupted or hash-invalid checkpoint is automatically evicted from `completed_shards` so only that shard is re-executed.

---

## 6. Stage/Shard Resumability & Recovery

Jobs are partitioned into deterministic work units/shards:
- **Completed Shards**: Listed in `completed_shards` in `manifest.json`. After a Space restart, completed shards are **never regenerated**.
- **Partially Completed Shards**: Resumed from the latest usable checkpoint if supported; otherwise, only that specific work unit is restarted.
- **Incompatible Resumes**: Continuation is automatically blocked if `source_repo`, `source_revision`, `source_path`, `source_split`, `configuration_hash`, `prompt_config_hash`, `output_schema_version`, or `code_revision` materially changed.
- **Forking Incompatible Jobs**: Use `POST /jobs/{job_id}/fork` to spawn a new durable job with modified parameters rather than mixing incompatible outputs.

---

## 7. Work Loss Estimates & Non-Resumable Boundaries

### Granularity & Resumability Boundaries
- **Fully Restart-Safe**: Stage boundaries, completed shards, and verified checkpoints.
- **Within a Work Unit**: Individual LLM API calls within an active uncheckpointed batch/shard are executed concurrently. If a hard Space termination occurs mid-shard, in-flight API calls that were not yet committed to a confirmed checkpoint will be re-executed when that shard restarts.

### Maximum Expected Work Lost
- **Worst-case loss on Space crash**: At most **1 active shard** or **1 micro-checkpoint interval** (e.g. ~10 minutes or 100 records, depending on configured policy). All previously verified shards remain fully intact in `HF_WORK_REPO`.

---

## 8. Launching and Managing Jobs via API

### Launch a New Pipeline Job
```bash
curl -X POST "https://<your-space>.hf.space/pipelines/run" \
     -H "Content-Type: application/json" \
     -d '{
           "node_path": "factual-datagen-pipeline",
           "config_path": "external:_START_HERE_complete_factual.yaml",
           "parameters": {
             "job_id": "custom_job_001",
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

### Fork a Job for Modified Parameters
```bash
curl -X POST "https://<your-space>.hf.space/jobs/<job_id>/fork?new_job_id=custom_fork_001" \
     -H "Content-Type: application/json" \
     -d '{
           "source_revision": "v2.0"
         }'
```
