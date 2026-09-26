# tasks.py
import os
import sys
import time
import random
import logging
import subprocess
import json
import signal
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
    HFHubError,
)

if TYPE_CHECKING:
    from huey.api import Task

PIPELINE_RUNNER_SCRIPT = "run_augmentoolkit.py"
ATK3_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ATK3_DIRECTORY, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PID_KEY_TIMEOUT = 24 * 60 * 60  # 24 hours
OUTPUT_DIR_MAPPING_TIMEOUT = PID_KEY_TIMEOUT * 7
PARAMETERS_TIMEOUT = OUTPUT_DIR_MAPPING_TIMEOUT
FINAL_STATUS_TIMEOUT = OUTPUT_DIR_MAPPING_TIMEOUT


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


@huey.task(context=True)
def run_pipeline_task(
    task: "Task",
    node_path: str,
    config_path: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
):
    """
    Executes a pipeline via a subprocess, manages Hugging Face Hub persistence (source, work, output repos),
    stores subprocess PID and parameters, and records durable job manifests and checkpoints.
    """
    task_id = str(task.id)
    redis_pid_key = f"worker_pid_for_task:{task_id}"
    redis_output_dir_key = f"output_dir_for_task:{task_id}"
    redis_params_key = f"parameters_for_task:{task_id}"
    redis_status_key = f"status_for_task:{task_id}"
    process = None
    log_file: Optional[io.TextIOWrapper] = None
    log_file_path = ""
    final_status_set = False

    os.environ["JOB_ID"] = task_id
    os.environ["TASK_ID"] = task_id

    print(f"Task {task_id}: Preparing to run pipeline subprocess for node: {node_path}")

    try:
        set_progress(task_id, 0.0, "Initializing task...")
    except Exception as e:
        print(f"Task {task_id}: Failed to set initial progress: {e}")

    if parameters is None:
        parameters = {}

    no_flatten_keys = parameters.get("no_flatten", [])
    try:
        parameters_flat = flatten_config(parameters, no_flatten_keys=no_flatten_keys)
    except Exception as fc_e:
        set_final_status(
            task_id,
            "FAILED",
            f"Task failed during parameter flattening: {fc_e}",
            details={"error": str(fc_e), "traceback": traceback.format_exc()},
        )
        raise

    if "task_id" not in parameters_flat:
        parameters_flat["task_id"] = task_id

    # --- HF Hub Repos & Source Handling ---
    hf_source_repo = parameters_flat.get("hf_source_repo") or os.environ.get("HF_SOURCE_REPO", "")
    hf_work_repo = parameters_flat.get("hf_work_repo") or os.environ.get("HF_WORK_REPO", "")
    hf_output_repo = parameters_flat.get("hf_output_repo") or os.environ.get("HF_OUTPUT_REPO", "")

    source_revision = parameters_flat.get("source_revision", "main")
    source_path = parameters_flat.get("source_path")
    source_split = parameters_flat.get("source_split")

    ckpt_mgr = CheckpointManager()

    # Pull source subset from HF_SOURCE_REPO if configured
    if hf_source_repo and source_path:
        try:
            print(f"Task {task_id}: Downloading source path '{source_path}' from {hf_source_repo} (revision: {source_revision})...")
            local_inputs_target = os.path.join(ATK3_DIRECTORY, "inputs", source_path)
            ckpt_mgr.hf_manager.download_folder(
                path_in_repo=source_path,
                local_dir=local_inputs_target,
                repo_id=hf_source_repo,
                revision=source_revision,
            )
            print(f"Task {task_id}: Source download complete.")
        except Exception as src_e:
            print(f"Task {task_id}: Failed to download source from {hf_source_repo}: {src_e}")
            set_final_status(
                task_id,
                "FAILED",
                f"Failed to download source input from HF Hub: {src_e}",
                details={"error": str(src_e)},
            )
            raise

    # Initialize remote JobManifest
    manifest: Optional[JobManifest] = None
    if hf_work_repo:
        os.environ["HF_WORK_REPO"] = hf_work_repo
        try:
            manifest = ckpt_mgr.init_job_manifest(
                job_id=task_id,
                pipeline=node_path,
                source_repo=hf_source_repo,
                source_revision=source_revision,
                source_path=source_path,
                source_split=source_split,
                work_repo=hf_work_repo,
                output_repo=hf_output_repo,
                config_dict=parameters_flat,
                sync_remote=True,
            )
            manifest.set_status("RUNNING")
            ckpt_mgr.save_manifest_remote(manifest)
            print(f"Task {task_id}: Initialized and uploaded remote job manifest to {hf_work_repo}")
        except Exception as manifest_e:
            print(f"Task {task_id}: Warning: Failed to sync initial manifest: {manifest_e}")

    try:
        # --- Determine and Store Output Directory ---
        output_dir_value = None
        resolved_output_dir = None

        with open("super_config.yaml", "r", encoding="utf-8") as f:
            super_config = yaml.safe_load(f)
        path_aliases = super_config.get("path_aliases", {})

        if "output_dir" in parameters_flat:
            output_dir_value = parameters_flat["output_dir"]
        elif config_path:
            try:
                resolved_config_from_alias = resolve_path(config_path, path_aliases)
                abs_config_path = Path(ATK3_DIRECTORY) / resolved_config_from_alias
                if abs_config_path.is_file():
                    with open(abs_config_path, "r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f)
                    if config_data:
                        output_dir_value = find_first_output_dir(config_data)
            except Exception as e:
                print(f"Task {task_id}: Error resolving config path output_dir: {e}")

        if output_dir_value and isinstance(output_dir_value, str):
            output_dir_path = Path(output_dir_value)
            resolved_output_dir = output_dir_path if output_dir_path.is_absolute() else (Path(ATK3_DIRECTORY) / output_dir_value).resolve()
        else:
            resolved_output_dir = (Path(ATK3_DIRECTORY) / "outputs" / task_id).resolve()
            resolved_output_dir.mkdir(parents=True, exist_ok=True)
            parameters_flat["output_dir"] = str(resolved_output_dir)

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
            )
        else:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=ATK3_DIRECTORY,
            )

        if process is None:
            raise RuntimeError("Subprocess.Popen failed, process object is None.")

        subprocess_pid = process.pid
        redis_client.set(redis_pid_key, subprocess_pid, ex=PID_KEY_TIMEOUT)

        exit_code = process.wait()

        # Sync log file remotely
        if manifest and log_file_path:
            ckpt_mgr.sync_log_file(manifest, log_file_path)

        if exit_code == 0:
            set_final_status(task_id, "COMPLETED", f"Pipeline task {task_id} completed successfully.")
            final_status_set = True

            # Finalize output dataset upload to HF_OUTPUT_REPO
            if manifest and resolved_output_dir and os.path.exists(resolved_output_dir):
                final_items = []
                for root, _, files in os.walk(resolved_output_dir):
                    for fname in files:
                        if fname.endswith(".jsonl") or fname.endswith(".json"):
                            fpath = os.path.join(root, fname)
                            try:
                                with open(fpath, "r", encoding="utf-8") as f:
                                    if fname.endswith(".jsonl"):
                                        for line in f:
                                            if line.strip():
                                                final_items.append(json.loads(line))
                                    else:
                                        data = json.load(f)
                                        if isinstance(data, list):
                                            final_items.extend(data)
                                        elif isinstance(data, dict):
                                            final_items.append(data)
                            except Exception:
                                pass
                if final_items:
                    out_rev = ckpt_mgr.finalize_job(
                        manifest=manifest,
                        final_dataset_items=final_items,
                        format_type="jsonl",
                        sync_remote=True,
                        log_file_path=log_file_path,
                    )
                    print(f"Task {task_id}: Uploaded final dataset to {manifest.output_repo} (Commit: {out_rev})")

            return {
                "status": "success",
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

            error_message = f"Task {task_id}: Subprocess failed with exit code {exit_code}."
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
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass

        redis_client.delete(redis_pid_key)
