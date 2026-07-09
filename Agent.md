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
- `/health`: Returns 200 OK when the app is ready.
- `/api-docs`: Redirects to Swagger UI (`/docs`).

## Functional Endpoints
The following endpoints are exposed by the FastAPI backend and documented in `/api-docs`:
- `/pipelines/run`: Queue a dataset generation pipeline.
- `/pipelines/available`: Get available pipeline aliases.
- `/configs/aliases`: Get available config aliases.
- `/tasks/queue`: Get list of pending and scheduled tasks.
- `/tasks/{task_id}/status`: Get status of a pipeline run.
- `/tasks/{task_id}/parameters`: Get parameters used for a task.
- `/tasks/{task_id}/interrupt`: Interrupt or revoke a task.
- `/tasks/{task_id}/logs`: Get logs for a specific task.
- `/logs`: List or clear all logs.
- `/outputs/...`: Manage and download outputs.
- `/inputs/...`: Manage and upload inputs.
- `/configs/...`: Manage and duplicate configuration files.

## Deployment Workflow
To redeploy, use the following command:
```bash
hf upload Leon4gr45/augmentanonymous --repo-type=space
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
1. **Unified Port:** Both the React frontend and FastAPI backend run on port 7860. The backend serves the frontend static files.
2. **Redis Integration:** Deployment requires a Redis server. The Dockerfile starts `redis-server` in the background.
3. **Huey Worker:** A Huey worker is started alongside the API to handle long-running pipeline tasks asynchronously.
4. **Environment Variables:** Ensure `REDIS_HOST` and `REDIS_PORT` are set correctly if using an external Redis, otherwise they default to `localhost:6379`.
