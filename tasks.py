# tasks.py
import os
import sys
import time
import random
import logging
import subprocess
import json
import signal
import threading
from typing import Optional, Dict, Any, TYPE_CHECKING, IO, Union
import traceback
import io
import yaml
from pathlib import Path

from huey_config import huey
from redis_config import redis_client, set_progress
from resolve_path import resolve_path
from run_augmentoolkit import flatten_config

from augmentoolkit.hf_persistence import (
    HFHubManager,
    CheckpointManager,
    JobManifest,
    JobManifestManager,
    HFHubError,
)

if TYPE_CHECKING:
    from huey.api import Task

PIPELINE_RUNNER_SCRIPT = "run_augmentoolkit.py"
ATK3_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SUPER_CONFIG_PATH = os.path.join(ATK3_DIRECTORY, "super_config.yaml")
LOGS_DIR = os.path.join(ATK3_DIRECTORY, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PID_KEY_TIMEOUT = 24 * 60 * 60  # 24 hours
OUTPUT_DIR_MAPPING_TIMEOUT = PID_KEY_TIMEOUT * 7
PARAMETERS_TIMEOUT = OUTPUT_DIR_MAPPING_TIMEOUT
FINAL_STATUS_TIMEOUT = OUTPUT_DIR_MAPPING_TIMEOUT
JOB_LOCK_TIMEOUT = 12 * 60 * 60  # 12 hours

LUA_RELEASE_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


def find_first_output_dir(
    data: Union[Dict, list], key_name: str = "output_dir"
) -> Optional[str]:
    """Recursively searches dict/list for the first value associated with key_name."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == key_name:
                if isinstance(value, str):
                    return value
            elif isinstance(value, (dict, list)):
                found = find_first_output_dir(value, key_name)
                if found is not None:
                    return found
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                found = find_first_output_dir(item, key_name)
                if found is not None:
                    return found
    return None


def resolve_job_output_dir(
    node_path: str,
    config_path: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    durable_job_id: Optional[str] = None,
) -> str:
    """Resolves the exact absolute output_dir for a job based on parameters or pipeline config."""
    params = parameters or {}
    output_dir_value = None

    if "output_dir" in params:
        output_dir_value = params["output_dir"]
    elif config_path:
        try:
            if os.path.exists(SUPER_CONFIG_PATH):
                with open(SUPER_CONFIG_PATH, "r", encoding="utf-8") as f:
                    super_config = yaml.safe_load(f)
                path_aliases = super_config.get("path_aliases", {})
                resolved_config_from_alias = resolve_path(config_path, path_aliases)
                abs_config_path = Path(ATK3_DIRECTORY) / resolved_config_from_alias
                if abs_config_path.is_file():
                    with open(abs_config_path, "r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f)
                    if config_data:
                        output_dir_value = find_first_output_dir(config_data)
        except Exception as e:
            logger.warning(f"Error resolving config output_dir: {e}")

    if output_dir_value and isinstance(output_dir_value, str):
        output_dir_path = Path(output_dir_value)
        return str(output_dir_path if output_dir_path.is_absolute() else (Path(ATK3_DIRECTORY) / output_dir_value).resolve())

    job_id_str = durable_job_id or params.get("job_id") or params.get("task_id") or "default_job"
    fallback_dir = (Path(ATK3_DIRECTORY) / "outputs" / job_id_str).resolve()
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return str(fallback_dir)


def set_final_status(
    task_id: str, status: str, message: str, details: Optional[Dict] = None
):
    """Sets the final status of a task in Redis."""
    redis_key = f"status_for_task:{task_id}"
    status_data = {
        "status": status,
        "message": message,
        "details": details or {},
        "timestamp": time.time(),
    }
    try:
        redis_client.set(redis_key, json.dumps(status_data), ex=FINAL_STATUS_TIMEOUT)
        print(f"Task {task_id}: Set final status in Redis ({redis_key}) to {status}. Message: {message}")
    except Exception as e:
        print(f"Task {task_id}: Failed to set final status '{status}' in Redis ({redis_key}): {e}")


def release_owned_job_lock(job_id: str, task_id: str) -> bool:
    """Releases Redis lock for durable job_id ONLY if lock value matches current task_id."""
    lock_key = f"durable_job_lock:{job_id}"
    try:
        res = redis_client.eval(LUA_RELEASE_LOCK, 1, lock_key, task_id)
        return bool(res)
    except Exception as e:
        logger.warning(f"Error releasing lock for job {job_id}: {e}")
        return False


class JobLockHeartbeat:
    """Periodically renews Redis lease for long-running jobs if lock is still owned by task_id."""

    def __init__(self, job_id: str, task_id: str, lease_seconds: int = JOB_LOCK_TIMEOUT, interval_seconds: int = 60):
        self.job_id = job_id
        self.task_id = task_id
        self.lock_key = f"durable_job_lock:{job_id}"
        self.lease_ms = lease_seconds * 1000
        self.interval_seconds = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_heartbeat, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_heartbeat(self):
        while self._running:
            time.sleep(self.interval_seconds)
            if not self._running:
                break
            try:
                curr_owner = redis_client.get(self.lock_key)
                if curr_owner and curr_owner.decode("utf-8") == self.task_id:
                    redis_client.pexpire(self.lock_key, self.lease_ms)
                else:
                    logger.warning(f"Heartbeat lost lock ownership for job {self.job_id}")
                    break
            except Exception as e:
                logger.warning(f"Heartbeat error for job {self.job_id}: {e}")


@huey.task(context=True)
def run_pipeline_task(
    task: "Task",
    node_path: str,
    config_path: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
):
    """
    Executes a pipeline via a subprocess with explicit subprocess environment context,
    managing separate durable job_id and ephemeral Huey task_id with atomic Redis lock ownership,
    fail-closed remote persistence, and incremental sharded dataset uploading.
    """
    task_id = str(task.id)
    if parameters is None:
        parameters = {}

    durable_job_id = str(parameters.get("job_id") or parameters.get("task_id") or task_id)

    redis_pid_key = f"worker_pid_for_task:{task_id}"
    redis_output_dir_key = f"output_dir_for_task:{task_id}"
    redis_params_key = f"parameters_for_task:{task_id}"
    redis_status_key = f"status_for_task:{task_id}"
    job_lock_key = f"durable_job_lock:{durable_job_id}"

    # Atomic owned lock acquisition via SET NX PX
    lock_acquired = redis_client.set(job_lock_key, task_id, nx=True, px=JOB_LOCK_TIMEOUT * 1000)
    if not lock_acquired:
        curr_owner = redis_client.get(job_lock_key)
        owner_str = curr_owner.decode("utf-8") if curr_owner else "unknown"
        if owner_str != task_id:
            msg = f"Job '{durable_job_id}' is currently locked/executing under task ID '{owner_str}'. Duplicate execution rejected."
            logger.error(msg)
            set_final_status(task_id, "FAILED", msg, details={"error": "duplicate_job_execution"})
            raise RuntimeError(msg)

    heartbeat = JobLockHeartbeat(durable_job_id, task_id, lease_seconds=JOB_LOCK_TIMEOUT)
    heartbeat.start()

    process = None
    log_file: Optional[io.TextIOWrapper] = None
    log_file_path = ""
    final_status_set = False

    print(f"Task {task_id} (Durable Job ID: {durable_job_id}): Preparing pipeline subprocess for node: {node_path}")

    try:
        set_progress(task_id, 0.0, f"Initializing task for job {durable_job_id}...")
    except Exception as e:
        print(f"Task {task_id}: Failed to set initial progress: {e}")

    no_flatten_keys = parameters.get("no_flatten", [])
    try:
        parameters_flat = flatten_config(parameters, no_flatten_keys=no_flatten_keys)
    except Exception as fc_e:
        heartbeat.stop()
        release_owned_job_lock(durable_job_id, task_id)
        set_final_status(
            task_id,
            "FAILED",
            f"Task failed during parameter flattening: {fc_e}",
            details={"error": str(fc_e), "traceback": traceback.format_exc()},
        )
        raise

    parameters_flat["task_id"] = task_id
    parameters_flat["job_id"] = durable_job_id

    resolved_output_dir = resolve_job_output_dir(
        node_path=node_path,
        config_path=config_path,
        parameters=parameters_flat,
        durable_job_id=durable_job_id,
    )
    parameters_flat["output_dir"] = resolved_output_dir

    hf_source_repo = parameters_flat.get("hf_source_repo") or os.environ.get("HF_SOURCE_REPO", "")
    hf_work_repo = parameters_flat.get("hf_work_repo") or os.environ.get("HF_WORK_REPO", "")
    hf_output_repo = parameters_flat.get("hf_output_repo") or os.environ.get("HF_OUTPUT_REPO", "")

    source_revision = parameters_flat.get("source_revision", "main")
    source_path = parameters_flat.get("source_path")
    source_split = parameters_flat.get("source_split")

    # Compute live prompt hash on initial job creation
    path_aliases = {}
    if os.path.exists(SUPER_CONFIG_PATH):
        try:
            with open(SUPER_CONFIG_PATH, "r", encoding="utf-8") as f:
                path_aliases = yaml.safe_load(f).get("path_aliases", {})
        except Exception:
            pass

    resolved_node_path = resolve_path(node_path, path_aliases) if path_aliases else node_path
    pipeline_dir = os.path.dirname(resolved_node_path) if "/" in resolved_node_path else None

    prompt_dir = parameters_flat.get("prompt_folder")
    if not prompt_dir and pipeline_dir:
        candidate_pdir = os.path.join(ATK3_DIRECTORY, pipeline_dir, "prompts")
        if os.path.exists(candidate_pdir):
            prompt_dir = candidate_pdir

    default_prompt_dir = parameters_flat.get("default_prompt_folder")

    live_prompt_hash = JobManifestManager.compute_live_prompt_hash(
        prompt_folder=prompt_dir,
        default_prompt_folder=default_prompt_dir,
    )

    ckpt_mgr = CheckpointManager()

    # Pull source subset from HF_SOURCE_REPO if configured
    if hf_source_repo and source_path:
        try:
            print(f"Job {durable_job_id}: Downloading source path '{source_path}' from {hf_source_repo} (revision: {source_revision})...")
            local_inputs_target = os.path.join(ATK3_DIRECTORY, "inputs", source_path)
            ckpt_mgr.hf_manager.download_folder(
                path_in_repo=source_path,
                local_dir=local_inputs_target,
                repo_id=hf_source_repo,
                revision=source_revision,
            )
            print(f"Job {durable_job_id}: Source download complete.")
        except Exception as src_e:
            heartbeat.stop()
            release_owned_job_lock(durable_job_id, task_id)
            msg = f"Failed to download source input from HF Hub for job {durable_job_id}: {src_e}"
            logger.error(msg)
            set_final_status(task_id, "FAILED", msg, details={"error": str(src_e)})
            raise HFHubError(msg) from src_e

    # Initialize/Recover remote JobManifest and hydrate remote checkpoints into output_dir!
    manifest: Optional[JobManifest] = None
    if hf_work_repo:
        try:
            existing_manifest = ckpt_mgr.load_remote_manifest(job_id=durable_job_id, work_repo=hf_work_repo)
            if existing_manifest:
                # Reconcile and hydrate checkpoints into local resolved_output_dir
                reconciled_manifest, valid_shards, restored_data = ckpt_mgr.reconcile_and_resume_job(
                    job_id=durable_job_id,
                    target_pipeline=node_path,
                    target_config_dict=parameters_flat,
                    target_prompt_hash=live_prompt_hash,
                    target_source_repo=hf_source_repo,
                    target_source_revision=source_revision,
                    target_source_path=source_path,
                    target_source_split=source_split,
                    work_repo=hf_work_repo,
                    output_dir=resolved_output_dir,
                )
                manifest = reconciled_manifest
                manifest.assign_task_id(task_id)
                manifest.set_status("RUNNING")
            else:
                manifest = ckpt_mgr.init_job_manifest(
                    job_id=durable_job_id,
                    pipeline=node_path,
                    source_repo=hf_source_repo,
                    source_revision=source_revision,
                    source_path=source_path,
                    source_split=source_split,
                    work_repo=hf_work_repo,
                    output_repo=hf_output_repo,
                    config_dict=parameters_flat,
                    prompt_config_hash=live_prompt_hash,
                    sync_remote=False,
                )
                manifest.assign_task_id(task_id)
                manifest.set_status("RUNNING")

            ckpt_mgr.save_manifest_remote(manifest)
            print(f"Job {durable_job_id}: Manifest initialized/updated and hydrated into '{resolved_output_dir}'")
        except Exception as manifest_e:
            heartbeat.stop()
            release_owned_job_lock(durable_job_id, task_id)
            msg = f"Fail-closed: Manifest creation or remote sync failed for job {durable_job_id} on repo {hf_work_repo}: {manifest_e}"
            logger.error(msg)
            set_final_status(task_id, "FAILED", msg, details={"error": str(manifest_e)})
            raise HFHubError(msg) from manifest_e

    try:
        if resolved_output_dir:
            try:
                redis_client.set(redis_output_dir_key, str(resolved_output_dir), ex=OUTPUT_DIR_MAPPING_TIMEOUT)
            except Exception as e:
                print(f"Task {task_id}: Failed to store output dir mapping: {e}")

        # Store Parameters
        try:
            params_json = json.dumps(parameters_flat)
            redis_client.set(redis_params_key, params_json, ex=PARAMETERS_TIMEOUT)
        except Exception as e:
            print(f"Task {task_id}: Failed to store parameters in Redis: {e}")

        # Construct explicit environment dict for subprocess
        sub_env = os.environ.copy()
        sub_env["JOB_ID"] = durable_job_id
        sub_env["TASK_ID"] = task_id
        sub_env["HF_WORK_REPO"] = hf_work_repo
        sub_env["HF_SOURCE_REPO"] = hf_source_repo
        sub_env["HF_OUTPUT_REPO"] = hf_output_repo

        # Prepare Subprocess Command
        command = [
            sys.executable,
            PIPELINE_RUNNER_SCRIPT,
            "--node",
            node_path,
        ]
        if config_path:
            command.extend(["--config", config_path])

        if parameters_flat:
            params_json_for_command = json.dumps(parameters_flat)
            command.extend(["--override-json", params_json_for_command])

        log_file_path = os.path.join(LOGS_DIR, f"{task_id}.log")
        try:
            log_file = open(log_file_path, "w", encoding="utf-8")
        except IOError:
            log_file = None

        if log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=log_file,
                cwd=ATK3_DIRECTORY,
                env=sub_env,
            )
        else:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=ATK3_DIRECTORY,
                env=sub_env,
            )

        if process is None:
            raise RuntimeError("Subprocess.Popen failed, process object is None.")

        subprocess_pid = process.pid
        redis_client.set(redis_pid_key, subprocess_pid, ex=PID_KEY_TIMEOUT)

        exit_code = process.wait()

        # Sync log file remotely
        if manifest and log_file_path and os.path.exists(log_file_path):
            ckpt_mgr.sync_log_file(manifest, log_file_path)

        if exit_code == 0:
            # Incremental sharded dataset upload to HF_OUTPUT_REPO before setting COMPLETED
            if manifest and resolved_output_dir and os.path.exists(resolved_output_dir):
                out_rev = ckpt_mgr.finalize_job_sharded(
                    manifest=manifest,
                    output_dir=str(resolved_output_dir),
                    sync_remote=True,
                    log_file_path=log_file_path,
                )
                print(f"Job {durable_job_id}: Incremental sharded finalization uploaded to {manifest.output_repo} (Revision: {out_rev})")

            # Set COMPLETED status in Redis ONLY AFTER HF output finalization and manifest upload succeed!
            set_final_status(task_id, "COMPLETED", f"Pipeline job {durable_job_id} (task {task_id}) completed successfully.")
            final_status_set = True

            return {
                "status": "success",
                "job_id": durable_job_id,
                "task_id": task_id,
                "message": f"Pipeline {node_path} completed successfully.",
            }
        else:
            if manifest:
                manifest.set_status("FAILED", error_msg=f"Subprocess exit code {exit_code}")
                ckpt_mgr.save_manifest_remote(manifest)

            redis_status_key = f"status_for_task:{task_id}"
            final_status_json = redis_client.get(redis_status_key)
            if final_status_json:
                try:
                    status_data = json.loads(final_status_json)
                    if status_data.get("status") == "REVOKED":
                        return
                except Exception:
                    pass

            error_message = f"Job {durable_job_id} (task {task_id}): Subprocess failed with exit code {exit_code}."
            if not final_status_set:
                set_final_status(
                    task_id,
                    "FAILED",
                    f"Pipeline subprocess failed with exit code {exit_code}.",
                    details={"exit_code": exit_code},
                )
                final_status_set = True
            raise RuntimeError(f"Pipeline subprocess failed with exit code {exit_code}")

    except Exception as e:
        if manifest:
            manifest.set_status("FAILED", error_msg=str(e))
            try:
                ckpt_mgr.save_manifest_remote(manifest)
            except Exception:
                pass

        if not final_status_set:
            set_final_status(
                task_id,
                "FAILED",
                f"Task failed due to an error: {e}",
                details={"error": str(e), "traceback": traceback.format_exc()},
            )
            final_status_set = True

        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()

        raise

    finally:
        heartbeat.stop()
        release_owned_job_lock(durable_job_id, task_id)

        if log_file:
            try:
                log_file.close()
            except Exception:
                pass

        redis_client.delete(redis_pid_key)
