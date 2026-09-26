import json
import logging
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .hf_manager import HFHubError, HFHubManager
from .job_manifest import (
    IncompatibleResumeError,
    JobManifest,
    JobManifestManager,
    compute_file_hash,
    compute_hash,
)

logger = logging.getLogger(__name__)


class CheckpointPolicy:
    """Configurable checkpoint policy abstraction."""

    def __init__(
        self,
        shard_checkpoint_mandatory: bool = True,
        record_interval: int = 100,
        time_interval_minutes: float = 10.0,
    ):
        self.shard_checkpoint_mandatory = shard_checkpoint_mandatory
        self.record_interval = record_interval
        self.time_interval_seconds = time_interval_minutes * 60.0
        self.last_checkpoint_time = time.time()
        self.record_counter = 0

    def should_micro_checkpoint(self, records_added: int = 1) -> bool:
        self.record_counter += records_added
        elapsed = time.time() - self.last_checkpoint_time
        if self.record_counter >= self.record_interval or elapsed >= self.time_interval_seconds:
            self.record_counter = 0
            self.last_checkpoint_time = time.time()
            return True
        return False


class CheckpointManager:
    """Coordinates remote-first checkpoint persistence, corruption verification, stage/shard tracking, and job recovery."""

    def __init__(
        self,
        hf_manager: Optional[HFHubManager] = None,
        checkpoint_dir: str = "checkpoints",
        outputs_dir: str = "outputs",
        policy: Optional[CheckpointPolicy] = None,
    ):
        self.hf_manager = hf_manager or HFHubManager()
        self.checkpoint_dir = checkpoint_dir
        self.outputs_dir = outputs_dir
        self.policy = policy or CheckpointPolicy()
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

    def init_job_manifest(
        self,
        job_id: str,
        pipeline: str,
        source_repo: Optional[str] = None,
        source_revision: str = "main",
        source_path: Optional[str] = None,
        source_split: Optional[str] = None,
        work_repo: Optional[str] = None,
        output_repo: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        prompt_config_hash: str = "",
        output_schema_version: str = "1.0",
        sync_remote: bool = True,
    ) -> JobManifest:
        src_repo = source_repo or self.hf_manager.source_repo
        wk_repo = work_repo or self.hf_manager.work_repo
        out_repo = output_repo or self.hf_manager.output_repo

        config_hash = compute_hash(config_dict or {})

        manifest = JobManifest(
            job_id=job_id,
            pipeline=pipeline,
            source_repo=src_repo,
            source_revision=source_revision,
            source_path=source_path,
            source_split=source_split,
            work_repo=wk_repo,
            output_repo=out_repo,
            configuration_hash=config_hash,
            prompt_config_hash=prompt_config_hash,
            output_schema_version=output_schema_version,
            status="PENDING",
        )

        if sync_remote and wk_repo:
            self.save_manifest_remote(manifest)

        return manifest

    def save_manifest_remote(self, manifest: JobManifest) -> str:
        """Saves job manifest locally and uploads to HF_WORK_REPO. Fails closed if work_repo is set and upload fails."""
        manifest.update_timestamp()
        job_local_dir = os.path.join(self.checkpoint_dir, manifest.job_id)
        os.makedirs(job_local_dir, exist_ok=True)
        manifest_path = os.path.join(job_local_dir, "manifest.json")

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        if not manifest.work_repo:
            return "local_only"

        path_in_repo = f"jobs/{manifest.job_id}/manifest.json"
        try:
            commit_sha = self.hf_manager.upload_file(
                local_path=manifest_path,
                path_in_repo=path_in_repo,
                repo_id=manifest.work_repo,
                repo_type="dataset",
                commit_message=f"Update manifest for job {manifest.job_id} (Status: {manifest.status})",
            )
            return commit_sha
        except Exception as e:
            msg = f"Fail-closed: Manifest creation/sync failed for job {manifest.job_id} on {manifest.work_repo}: {e}"
            logger.error(msg)
            raise HFHubError(msg) from e

    def save_checkpoint(
        self,
        manifest: JobManifest,
        shard_id: Union[str, int],
        checkpoint_data: Dict[str, Any],
        stage: str = "generation",
        sync_remote: bool = True,
        log_file_path: Optional[str] = None,
    ) -> str:
        """
        Remote-first checkpoint save with checksum calculation and strict fail-closed verification:
        1. Writes local checkpoint JSON.
        2. Computes SHA256 checksum and size metadata.
        3. Uploads checkpoint file and updated manifest to HF_WORK_REPO.
        4. Verifies remote file existence.
        5. Deletes local temporary checkpoint file ONLY after confirmed remote persistence.
        """
        job_id = manifest.job_id
        manifest.current_stage = stage
        manifest.current_shard = shard_id

        local_job_dir = os.path.join(self.checkpoint_dir, job_id)
        os.makedirs(local_job_dir, exist_ok=True)

        shard_filename = f"shard_{shard_id}.json"
        local_ckpt_path = os.path.join(local_job_dir, shard_filename)

        with open(local_ckpt_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        file_size = os.path.getsize(local_ckpt_path)
        checksum = compute_file_hash(local_ckpt_path)
        record_count = len(checkpoint_data) if isinstance(checkpoint_data, (dict, list)) else 1

        manifest.mark_shard_completed(
            shard_id=shard_id,
            checksum=checksum,
            size=file_size,
            record_count=record_count,
        )

        if not sync_remote or not manifest.work_repo:
            logger.info(f"Checkpoint saved locally for job {job_id}, shard {shard_id}")
            return "local_saved"

        repo_ckpt_path = f"jobs/{job_id}/checkpoints/{shard_filename}"

        try:
            commit_sha = self.hf_manager.upload_file(
                local_path=local_ckpt_path,
                path_in_repo=repo_ckpt_path,
                repo_id=manifest.work_repo,
                repo_type="dataset",
                commit_message=f"Checkpoint job {job_id} stage {stage} shard {shard_id}",
            )

            if not self.hf_manager.file_exists(
                repo_id=manifest.work_repo, path_in_repo=repo_ckpt_path
            ):
                raise HFHubError(
                    f"Remote verification failed: Checkpoint {repo_ckpt_path} not found after upload."
                )

            self.save_manifest_remote(manifest)

            if log_file_path and os.path.exists(log_file_path):
                self.sync_log_file(manifest, log_file_path)

            try:
                os.remove(local_ckpt_path)
            except Exception as e:
                logger.warning(f"Could not remove local temp checkpoint {local_ckpt_path}: {e}")

            return commit_sha

        except Exception as e:
            msg = f"Remote checkpoint persistence failed closed for job {job_id}, shard {shard_id}: {e}"
            logger.error(msg)
            manifest.set_status("FAILED", error_msg=msg)
            try:
                self.save_manifest_remote(manifest)
            except Exception:
                pass
            raise HFHubError(msg) from e

    def sync_log_file(self, manifest: JobManifest, log_file_path: str, max_lines: int = 2000):
        if not manifest.work_repo or not os.path.exists(log_file_path):
            return

        job_id = manifest.job_id
        try:
            with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            bounded_lines = lines[-max_lines:] if len(lines) > max_lines else lines

            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".log", delete=False) as tmp:
                tmp.writelines(bounded_lines)
                tmp_path = tmp.name

            path_in_repo = f"jobs/{job_id}/logs/job.log"
            self.hf_manager.upload_file(
                local_path=tmp_path,
                path_in_repo=path_in_repo,
                repo_id=manifest.work_repo,
                repo_type="dataset",
                commit_message=f"Sync logs for job {job_id}",
            )
        except Exception as e:
            logger.warning(f"Failed to sync log file for job {job_id}: {e}")
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_remote_manifest(
        self, job_id: str, work_repo: Optional[str] = None
    ) -> Optional[JobManifest]:
        manifest_dict = self.hf_manager.fetch_manifest(job_id=job_id, work_repo=work_repo)
        if manifest_dict:
            return JobManifest.from_dict(manifest_dict)
        return None

    def reconcile_and_resume_job(
        self,
        job_id: str,
        target_pipeline: str,
        target_config_dict: Optional[Dict[str, Any]] = None,
        target_prompt_hash: str = "",
        target_source_repo: Optional[str] = None,
        target_source_revision: str = "main",
        target_source_path: Optional[str] = None,
        target_source_split: Optional[str] = None,
        target_output_schema_version: str = "1.0",
        work_repo: Optional[str] = None,
    ) -> Tuple[JobManifest, Set[Union[str, int]], Dict[str, Any]]:
        """
        Startup/Resume reconciliation with strict corruption safeguards:
        1. Fetches remote manifest.
        2. Validates compatibility against target parameters.
        3. Downloads and verifies checksum/JSON validity for each shard.
        4. Corrupted/unreadable/invalid shards are dropped from completed_shards so only those shards re-run.
        """
        existing_manifest = self.load_remote_manifest(job_id=job_id, work_repo=work_repo)
        if not existing_manifest:
            raise ValueError(f"Job manifest for job_id '{job_id}' not found in remote repository.")

        src_repo = target_source_repo or existing_manifest.source_repo

        target_manifest = JobManifest(
            job_id=job_id,
            pipeline=target_pipeline,
            source_repo=src_repo,
            source_revision=target_source_revision,
            source_path=target_source_path or existing_manifest.source_path,
            source_split=target_source_split or existing_manifest.source_split,
            work_repo=existing_manifest.work_repo,
            output_repo=existing_manifest.output_repo,
            configuration_hash=compute_hash(target_config_dict or {}),
            prompt_config_hash=target_prompt_hash,
            output_schema_version=target_output_schema_version,
        )

        is_compat, mismatches = JobManifestManager.check_compatibility(
            existing_manifest, target_manifest
        )
        if not is_compat:
            reason = "; ".join(mismatches)
            msg = f"Refusing automatic continuation for job {job_id}: Material inputs changed ({reason}). Use fork job API or start new run."
            logger.error(msg)
            raise IncompatibleResumeError(msg)

        valid_completed_shards: Set[Union[str, int]] = set()
        restored_data: Dict[str, Any] = {}

        job_local_dir = os.path.join(self.checkpoint_dir, job_id)
        os.makedirs(job_local_dir, exist_ok=True)

        for shard_id in list(existing_manifest.completed_shards):
            shard_filename = f"shard_{shard_id}.json"
            repo_ckpt_path = f"jobs/{job_id}/checkpoints/{shard_filename}"
            local_ckpt_path = os.path.join(job_local_dir, shard_filename)

            try:
                self.hf_manager.download_file(
                    path_in_repo=repo_ckpt_path,
                    local_path=local_ckpt_path,
                    repo_id=existing_manifest.work_repo,
                    repo_type="dataset",
                )

                # Checksum verification if available
                shard_meta = existing_manifest.checkpoints_metadata.get(str(shard_id), {})
                expected_checksum = shard_meta.get("checksum")
                if expected_checksum:
                    actual_checksum = compute_file_hash(local_ckpt_path)
                    if actual_checksum != expected_checksum:
                        logger.warning(
                            f"Corruption detected for job {job_id} shard {shard_id}: Checksum mismatch ({actual_checksum} != {expected_checksum}). Evicting shard to force re-execution."
                        )
                        continue

                with open(local_ckpt_path, "r", encoding="utf-8") as f:
                    shard_content = json.load(f)
                    if not isinstance(shard_content, dict):
                        logger.warning(f"Corruption detected for job {job_id} shard {shard_id}: content is not a dict. Evicting shard.")
                        continue
                    restored_data.update(shard_content)

                valid_completed_shards.add(shard_id)

            except Exception as e:
                logger.warning(f"Could not load valid checkpoint for shard {shard_id} of job {job_id}: {e}. Evicting shard from completed set.")

        # Update manifest completed_shards to contain only valid shards
        existing_manifest.completed_shards = list(valid_completed_shards)
        existing_manifest.status = "RUNNING"
        self.save_manifest_remote(existing_manifest)

        return existing_manifest, valid_completed_shards, restored_data

    def discover_incomplete_jobs(self, work_repo: Optional[str] = None) -> List[JobManifest]:
        job_ids = self.hf_manager.list_jobs(work_repo=work_repo)
        incomplete_manifests = []

        for jid in job_ids:
            manifest = self.load_remote_manifest(job_id=jid, work_repo=work_repo)
            if manifest and manifest.status in ("PENDING", "RUNNING", "INTERRUPTED"):
                incomplete_manifests.append(manifest)

        return incomplete_manifests

    def finalize_job_sharded(
        self,
        manifest: JobManifest,
        output_dir: str,
        format_type: str = "jsonl",
        sync_remote: bool = True,
        log_file_path: Optional[str] = None,
    ) -> str:
        """
        Incremental sharded finalization (NO whole dataset in RAM!):
        Scans output_dir for output files/shards, uploads each shard incrementally to HF_OUTPUT_REPO,
        records each shard in manifest.output_shards, and sets manifest status to COMPLETED.
        """
        job_id = manifest.job_id
        if not os.path.exists(output_dir):
            raise ValueError(f"Output directory '{output_dir}' does not exist.")

        latest_commit_sha = "local_only"

        for root, _, files in sorted(os.walk(output_dir)):
            for fname in sorted(files):
                if fname.endswith((".jsonl", ".parquet", ".json")):
                    fpath = os.path.join(root, fname)
                    rel_path = os.path.relpath(fpath, output_dir)
                    f_size = os.path.getsize(fpath)
                    f_hash = compute_file_hash(fpath)

                    # Count records without loading full file into RAM
                    record_count = 0
                    if fname.endswith(".jsonl"):
                        with open(fpath, "r", encoding="utf-8") as f:
                            record_count = sum(1 for line in f if line.strip())
                    else:
                        record_count = 1

                    repo_path = f"datasets/{job_id}/{rel_path}"

                    if sync_remote and manifest.output_repo:
                        commit_sha = self.hf_manager.upload_file(
                            local_path=fpath,
                            path_in_repo=repo_path,
                            repo_id=manifest.output_repo,
                            repo_type="dataset",
                            commit_message=f"Final output shard {rel_path} for job {job_id}",
                        )
                        latest_commit_sha = commit_sha
                        manifest.record_output_shard(
                            filename=rel_path,
                            record_count=record_count,
                            sha256=f_hash,
                            commit_sha=commit_sha,
                            path_in_repo=repo_path,
                        )

        manifest.final_output_revision = latest_commit_sha
        manifest.set_status("COMPLETED")
        manifest.current_stage = "completed"

        if sync_remote and manifest.work_repo:
            self.save_manifest_remote(manifest)

        if log_file_path and os.path.exists(log_file_path):
            self.sync_log_file(manifest, log_file_path)

        return latest_commit_sha
