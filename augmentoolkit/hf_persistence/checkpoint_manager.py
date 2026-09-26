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


def split_large_jsonl(
    fpath: str,
    max_bytes_per_shard: int = 50 * 1024 * 1024,
    max_records_per_shard: int = 5000,
) -> List[str]:
    """
    Streams a large .jsonl file line-by-line and splits it into deterministic bounded shard files
    if size or record count exceeds limits. Deletes original file and returns list of created part file paths.
    """
    if not fpath.endswith(".jsonl") or not os.path.exists(fpath):
        return [fpath]

    file_size = os.path.getsize(fpath)
    if file_size <= max_bytes_per_shard:
        return [fpath]

    created_parts = []
    base_dir = os.path.dirname(fpath)
    stem = os.path.basename(fpath)[:-6]  # strip .jsonl

    part_idx = 0
    current_records = 0
    current_bytes = 0
    current_part_path = os.path.join(base_dir, f"{stem}_part{part_idx:03d}.jsonl")
    current_file = open(current_part_path, "w", encoding="utf-8")
    created_parts.append(current_part_path)

    with open(fpath, "r", encoding="utf-8", errors="replace") as src:
        for line in src:
            line_bytes = len(line.encode("utf-8"))
            if (current_bytes + line_bytes > max_bytes_per_shard or current_records >= max_records_per_shard) and current_records > 0:
                current_file.close()
                part_idx += 1
                current_records = 0
                current_bytes = 0
                current_part_path = os.path.join(base_dir, f"{stem}_part{part_idx:03d}.jsonl")
                current_file = open(current_part_path, "w", encoding="utf-8")
                created_parts.append(current_part_path)

            current_file.write(line)
            current_records += 1
            current_bytes += line_bytes

    current_file.close()

    try:
        os.remove(fpath)
    except Exception as e:
        logger.warning(f"Could not remove original large file {fpath}: {e}")

    return created_parts


class CheckpointManager:
    """Coordinates remote-first checkpoint persistence, state hydration, corruption verification, stage/shard tracking, and job recovery."""

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

        config_hash = JobManifestManager.canonical_generation_spec_hash(config_dict or {})

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
        output_dir: Optional[str] = None,
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

        shard_str = str(shard_id)
        shard_filename = shard_str if shard_str.endswith(".json") else f"{shard_str}.json"
        local_ckpt_path = os.path.join(local_job_dir, shard_filename)

        with open(local_ckpt_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_ckpt_path = os.path.join(output_dir, shard_filename)
            with open(out_ckpt_path, "w", encoding="utf-8") as f:
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
        output_dir: Optional[str] = None,
    ) -> Tuple[JobManifest, Set[Union[str, int]], Dict[str, Any]]:
        """
        Startup/Resume reconciliation with strict corruption safeguards and local stage output hydration:
        1. Fetches remote manifest.
        2. Validates compatibility against target parameters.
        3. Downloads and verifies checksum/JSON validity for each shard.
        4. Hydrates valid checkpoint files directly into output_dir so PipelineStep.load_dataset() sees them.
        5. Corrupted/unreadable/invalid shards are dropped from completed_shards so only those shards re-run.
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
            configuration_hash=JobManifestManager.canonical_generation_spec_hash(target_config_dict or {}),
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
            shard_str = str(shard_id)
            shard_filename = shard_str if shard_str.endswith(".json") else f"{shard_str}.json"
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
                shard_meta = existing_manifest.checkpoints_metadata.get(shard_str, {})
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

                # Hydrate directly into target output_dir on local disk so PipelineStep.load_dataset() finds it
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    out_ckpt_path = os.path.join(output_dir, shard_filename)
                    with open(out_ckpt_path, "w", encoding="utf-8") as f:
                        json.dump(shard_content, f, indent=2)

                valid_completed_shards.add(shard_id)

            except Exception as e:
                logger.warning(f"Could not load valid checkpoint for shard {shard_id} of job {job_id}: {e}. Evicting shard from completed set.")

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
        max_bytes_per_shard: int = 50 * 1024 * 1024,  # 50MB max shard size
        max_records_per_shard: int = 5000,
    ) -> str:
        """
        Incremental sharded finalization (NO whole dataset in RAM!):
        1. Splits large .jsonl files line-by-line by size/record count before uploading.
           (Note: single-file upload is preserved for .json or .parquet files that cannot be safely line-split).
        2. Reconciles already uploaded shards recorded in manifest.output_shards and skips valid uploaded shards on retry.
        3. Uploads each un-uploaded shard immediately to HF_OUTPUT_REPO and records shard entry in work manifest immediately after each upload.
        """
        job_id = manifest.job_id
        if not os.path.exists(output_dir):
            raise ValueError(f"Output directory '{output_dir}' does not exist.")

        already_uploaded_paths = {s["path_in_repo"]: s["sha256"] for s in manifest.output_shards}
        latest_commit_sha = manifest.final_output_revision or "local_only"

        # First pass: split large .jsonl files if needed
        for root, _, files in sorted(os.walk(output_dir)):
            for fname in sorted(files):
                if fname.endswith(".jsonl") and not "_part" in fname:
                    fpath = os.path.join(root, fname)
                    split_large_jsonl(fpath, max_bytes_per_shard=max_bytes_per_shard, max_records_per_shard=max_records_per_shard)

        # Second pass: upload each shard
        for root, _, files in sorted(os.walk(output_dir)):
            for fname in sorted(files):
                if fname.endswith((".jsonl", ".parquet", ".json")):
                    fpath = os.path.join(root, fname)
                    rel_path = os.path.relpath(fpath, output_dir)
                    f_hash = compute_file_hash(fpath)
                    repo_path = f"datasets/{job_id}/{rel_path}"

                    # Reconcile already uploaded output shards by path/hash and skip valid ones
                    if repo_path in already_uploaded_paths and already_uploaded_paths[repo_path] == f_hash:
                        logger.info(f"Output shard {repo_path} already uploaded and verified. Skipping.")
                        continue

                    record_count = 0
                    if fname.endswith(".jsonl"):
                        with open(fpath, "r", encoding="utf-8") as f:
                            record_count = sum(1 for line in f if line.strip())
                    else:
                        record_count = 1

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
                        # Persist manifest immediately after EACH successfully verified output shard upload
                        self.save_manifest_remote(manifest)

        manifest.final_output_revision = latest_commit_sha
        manifest.set_status("COMPLETED")
        manifest.current_stage = "completed"

        if sync_remote and manifest.work_repo:
            self.save_manifest_remote(manifest)

        if log_file_path and os.path.exists(log_file_path):
            self.sync_log_file(manifest, log_file_path)

        return latest_commit_sha
