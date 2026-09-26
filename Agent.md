# Deployment Manager Information

This codebase is configured for deployment to Hugging Face Spaces.

## Target Space
- **Profile:** Leon4gr45
- **Space:** augmentanonymous
- **Identifier:** Leon4gr45/augmentanonymous
- **Port:** 7860

## Deployment Strategy
- **SDK:** Docker
- **Base Image:** python:3.11-slim
- **Port Mapping:** 7860 (FastAPI serving both API and React frontend)

## Mandatory Endpoints
- `/health`: Returns 200 OK when the app is ready (`{"status": "ok", "message": "Augmentoolkit API is running."}`).
- `/api-docs`: Redirects to Swagger UI (`/docs`).

## Functional Endpoints
The following endpoints are exposed by the FastAPI backend and documented in `/api-docs`:
- `/pipelines/run`: Queue a dataset generation pipeline.
- `/pipelines/available`: Get available pipeline aliases.
- `/configs/aliases`: Get available config aliases.
- `/jobs`: List jobs from HF_WORK_REPO.
- `/jobs/{job_id}/manifest`: Inspect job manifest.
- `/jobs/{job_id}/checkpoints`: Inspect job checkpoints.
- `/jobs/{job_id}/outputs`: Inspect job output references.
- `/jobs/{job_id}/resume`: Explicitly resume a job from remote state.
- `/jobs/{job_id}/fork`: Fork an existing job run into a new durable job.
- `/tasks/queue`: Get list of pending and scheduled tasks.
- `/tasks/{task_id}/status`: Get status of a pipeline run.
- `/tasks/{task_id}/parameters`: Get parameters used for a task.
- `/tasks/{task_id}/interrupt`: Interrupt or revoke a task.
- `/tasks/{task_id}/logs`: Get logs for a specific task.
- `/logs`: Clear all logs.
- `/outputs/...`: Manage and download outputs.
- `/inputs/...`: Manage and upload inputs.
- `/configs/...`: Manage and duplicate configuration files.

## Deployment Workflow
To redeploy, use the following command (with `--exclude "inputs/*" --exclude "outputs/*"` if inputs contains binary files):
```bash
hf upload Leon4gr45/augmentanonymous . --repo-type=space --exclude "inputs/*" --exclude "outputs/*"
```

## Monitoring
- **Build Logs:**
```bash
curl -N -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/spaces/Leon4gr45/augmentanonymous/logs/build"
```
- **Run Logs:**
```bash
curl -N -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/spaces/Leon4gr45/augmentanonymous/logs/run"
```

## Tricks and Best Practices
1. **Binary Files Exclusion:** When uploading to Hugging Face Spaces via `hf upload`, binary files in input or output directories (such as `.pdf.txt` files with PDF bytes) can trigger Xet storage errors. Use `--exclude "inputs/*"` or configure `.hfignore` appropriately.
2. **Unified Port:** Both the React frontend and FastAPI backend run on port 7860. The backend serves the frontend static files from `atk-interface/dist`.
3. **Redis Integration:** Deployment requires a Redis server. The Dockerfile starts `redis-server` in the background for task management and Huey queue support.
4. **Huey Worker:** A Huey worker is started alongside the API in `start.sh` to handle long-running pipeline tasks asynchronously.
5. **Space Domain Access:** On initial container boot, `/health` and `/api-docs` endpoints become accessible on `https://leon4gr45-augmentanonymous.hf.space` once the space transition state completes from `APP_STARTING` to `RUNNING`.
