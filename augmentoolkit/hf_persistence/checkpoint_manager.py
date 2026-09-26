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
    compute_hash,
)

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Coordinates remote-first checkpoint persistence, stage/shard tracking, and job recovery."""

    def __init__(
        self,
        hf_manager: Optional[HFHubManager] = None,
        checkpoint_dir: str = "checkpoints",
        outputs_dir: str = "outputs",
    ):
        self.hf_manager = hf_manager or HFHubManager()
        self.checkpoint_dir = checkpoint_dir
        self.outputs_dir = outputs_dir
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
        sync_remote: bool = True,
    ) -> JobManifest:
        """Creates and optionally persists a new job manifest."""
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
            status="PENDING",
        )

        if sync_remote and wk_repo:
            self.save_manifest_remote(manifest)

        return manifest

    def save_manifest_remote(self, manifest: JobManifest) -> str:
        """Saves job manifest locally and uploads to HF_WORK_REPO under jobs/{job_id}/manifest.json."""
        manifest.update_timestamp()
        job_local_dir = os.path.join(self.checkpoint_dir, manifest.job_id)
        os.makedirs(job_local_dir, exist_ok=True)
        manifest_path = os.path.join(job_local_dir, "manifest.json")

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        if not manifest.work_repo:
            logger.warning(f"No work_repo set for job {manifest.job_id}, manifest saved locally only.")
            return "local_only"

        path_in_repo = f"jobs/{manifest.job_id}/manifest.json"
        commit_sha = self.hf_manager.upload_file(
            local_path=manifest_path,
            path_in_repo=path_in_repo,
            repo_id=manifest.work_repo,
            repo_type="dataset",
            commit_message=f"Update manifest for job {manifest.job_id} (Status: {manifest.status})",
        )
        return commit_sha

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
        Remote-first checkpoint save:
        1. Writes local checkpoint file.
        2. Uploads checkpoint file and updated manifest to HF_WORK_REPO.
        3. Verifies successful remote upload before treating checkpoint as durable.
        4. Deletes local temporary checkpoint file if remote upload succeeds.
        Fails closed on remote upload failure.
        """
        job_id = manifest.job_id
        manifest.current_stage = stage
        manifest.current_shard = shard_id

        local_job_dir = os.path.join(self.checkpoint_dir, job_id)
        os.makedirs(local_job_dir, exist_ok=True)

        shard_filename = f"shard_{shard_id}.json"
        local_ckpt_path = os.path.join(local_job_dir, shard_filename)

        # Write local checkpoint
        with open(local_ckpt_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        if not sync_remote or not manifest.work_repo:
            logger.info(f"Checkpoint saved locally for job {job_id}, shard {shard_id}")
            manifest.mark_shard_completed(shard_id)
            return "local_saved"

        repo_ckpt_path = f"jobs/{job_id}/checkpoints/{shard_filename}"

        # Remote upload
        try:
            commit_sha = self.hf_manager.upload_file(
                local_path=local_ckpt_path,
                path_in_repo=repo_ckpt_path,
                repo_id=manifest.work_repo,
                repo_type="dataset",
                commit_message=f"Checkpoint job {job_id} stage {stage} shard {shard_id}",
            )

            # Verification: file must exist on hub
            if not self.hf_manager.file_exists(
                repo_id=manifest.work_repo, path_in_repo=repo_ckpt_path
            ):
                raise HFHubError(
                    f"Remote verification failed: Checkpoint {repo_ckpt_path} not found after upload."
                )

            # Mark completed in manifest and save updated manifest
            manifest.mark_shard_completed(shard_id)
            self.save_manifest_remote(manifest)

            # Sync logs if provided
            if log_file_path and os.path.exists(log_file_path):
                self.sync_log_file(manifest, log_file_path)

            # Delete local temp file only after confirmed remote persistence
            try:
                os.remove(local_ckpt_path)
                logger.info(f"Deleted local temporary checkpoint after remote verification: {local_ckpt_path}")
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
        """Uploads bounded/truncated log file to HF_WORK_REPO under jobs/{job_id}/logs/job.log."""
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
        """Downloads and constructs JobManifest from HF_WORK_REPO."""
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
        target_source_revision: str = "main",
        work_repo: Optional[str] = None,
    ) -> Tuple[JobManifest, Set[Union[str, int]], Dict[str, Any]]:
        """
        Startup/Resume reconciliation:
        1. Connects to work_repo, fetches remote manifest.
        2. Validates compatibility against target parameters/hashes.
        3. Returns (manifest, completed_shards_set, merged_checkpoint_data).
        Refuses continuation if incompatible.
        """
        existing_manifest = self.load_remote_manifest(job_id=job_id, work_repo=work_repo)
        if not existing_manifest:
            raise ValueError(f"Job manifest for job_id '{job_id}' not found in remote repository.")

        # Construct candidate target manifest to compare compatibility
        target_manifest = JobManifest(
            job_id=job_id,
            pipeline=target_pipeline,
            source_repo=existing_manifest.source_repo,
            source_revision=target_source_revision,
            work_repo=existing_manifest.work_repo,
            output_repo=existing_manifest.output_repo,
            configuration_hash=compute_hash(target_config_dict or {}),
            prompt_config_hash=target_prompt_hash,
        )

        is_compat, mismatches = JobManifestManager.check_compatibility(
            existing_manifest, target_manifest
        )
        if not is_compat:
            reason = "; ".join(mismatches)
            msg = f"Refusing automatic continuation for job {job_id}: Material inputs changed ({reason}). Required to start new or forked run."
            logger.error(msg)
            raise IncompatibleResumeError(msg)

        # Restore completed shard data
        completed_shards = set(existing_manifest.completed_shards)
        restored_data: Dict[str, Any] = {}

        # Download existing remote checkpoints into local checkpoint directory
        job_local_dir = os.path.join(self.checkpoint_dir, job_id)
        os.makedirs(job_local_dir, exist_ok=True)

        for shard_id in completed_shards:
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
                with open(local_ckpt_path, "r", encoding="utf-8") as f:
                    shard_content = json.load(f)
                    restored_data.update(shard_content)
            except Exception as e:
                logger.warning(f"Could not download completed shard {shard_id} for job {job_id}: {e}")

        existing_manifest.status = "RUNNING"
        self.save_manifest_remote(existing_manifest)

        return existing_manifest, completed_shards, restored_data

    def discover_incomplete_jobs(self, work_repo: Optional[str] = None) -> List[JobManifest]:
        """Finds all jobs in work_repo with status PENDING, RUNNING, or INTERRUPTED."""
        job_ids = self.hf_manager.list_jobs(work_repo=work_repo)
        incomplete_manifests = []

        for jid in job_ids:
            manifest = self.load_remote_manifest(job_id=jid, work_repo=work_repo)
            if manifest and manifest.status in ("PENDING", "RUNNING", "INTERRUPTED"):
                incomplete_manifests.append(manifest)

        return incomplete_manifests

    def finalize_job(
        self,
        manifest: JobManifest,
        final_dataset_items: List[Dict[str, Any]],
        format_type: str = "jsonl",
        sync_remote: bool = True,
        log_file_path: Optional[str] = None,
    ) -> str:
        """
        Writes final generated training data to HF_OUTPUT_REPO (as Parquet/JSONL)
        and records final output revision in the job manifest.
        """
        job_id = manifest.job_id
        job_out_dir = os.path.join(self.outputs_dir, job_id)
        os.makedirs(job_out_dir, exist_ok=True)

        if format_type.lower() == "jsonl":
            out_filename = "train.jsonl"
            out_local_path = os.path.join(job_out_dir, out_filename)
            with open(out_local_path, "w", encoding="utf-8") as f:
                for item in final_dataset_items:
                    f.write(json.dumps(item) + "\n")
        elif format_type.lower() == "parquet":
            import pandas as pd
            out_filename = "train.parquet"
            out_local_path = os.path.join(job_out_dir, out_filename)
            df = pd.DataFrame(final_dataset_items)
            df.to_parquet(out_local_path, index=False)
        else:
            out_filename = "data.json"
            out_local_path = os.path.join(job_out_dir, out_filename)
            with open(out_local_path, "w", encoding="utf-8") as f:
                json.dump(final_dataset_items, f, indent=2)

        output_commit_sha = "local_only"
        if sync_remote and manifest.output_repo:
            repo_path = f"datasets/{job_id}/{out_filename}"
            output_commit_sha = self.hf_manager.upload_file(
                local_path=out_local_path,
                path_in_repo=repo_path,
                repo_id=manifest.output_repo,
                repo_type="dataset",
                commit_message=f"Final output dataset for job {job_id}",
            )
            manifest.final_output_revision = output_commit_sha

        manifest.set_status("COMPLETED")
        manifest.current_stage = "completed"
        if sync_remote and manifest.work_repo:
            self.save_manifest_remote(manifest)

        if log_file_path and os.path.exists(log_file_path):
            self.sync_log_file(manifest, log_file_path)

        return output_commit_sha
