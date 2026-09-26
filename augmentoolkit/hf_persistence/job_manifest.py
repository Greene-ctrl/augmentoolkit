import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union


class IncompatibleResumeError(Exception):
    """Raised when attempting to resume a job with incompatible configurations or code revisions."""
    pass


def compute_hash(data: Any) -> str:
    """Computes SHA256 hash of arbitrary data (dict, string, bytes, etc.)."""
    if isinstance(data, (bytes, bytearray)):
        return hashlib.sha256(data).hexdigest()
    if isinstance(data, str):
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
    # Serialize dict/list with sorted keys for deterministic hashing
    canonical_json = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_dir_hash(dir_path: str) -> str:
    """Computes hash of all files in a directory deterministically."""
    if not dir_path or not os.path.exists(dir_path):
        return compute_hash(f"EMPTY_OR_NONEXISTENT:{dir_path}")

    hashes = []
    for root, _, files in sorted(os.walk(dir_path)):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    rel_path = os.path.relpath(fpath, dir_path)
                    hashes.append(f"{rel_path}:{file_hash}")
            except Exception:
                pass
    return compute_hash(hashes)


def get_code_revision() -> str:
    """Attempts to retrieve git commit SHA, falling back to version/env variable."""
    env_rev = os.environ.get("AUGMENTOOLKIT_CODE_REVISION")
    if env_rev:
        return env_rev
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "augmentoolkit-v1.0-hf-space"


class JobManifest:
    """Encapsulates the state and metadata of a dataset generation job."""

    def __init__(
        self,
        job_id: str,
        pipeline: str,
        source_repo: str,
        work_repo: str,
        output_repo: str,
        source_revision: str = "main",
        source_path: Optional[str] = None,
        source_split: Optional[str] = None,
        configuration_hash: str = "",
        prompt_config_hash: str = "",
        code_revision: Optional[str] = None,
        current_stage: str = "initialized",
        current_shard: Optional[Union[str, int]] = None,
        completed_shards: Optional[List[Union[str, int]]] = None,
        status: str = "PENDING",
        timestamps: Optional[Dict[str, Any]] = None,
        error_retry_info: Optional[Dict[str, Any]] = None,
        final_output_revision: Optional[str] = None,
    ):
        self.job_id = job_id
        self.pipeline = pipeline
        self.source_repo = source_repo
        self.source_revision = source_revision
        self.source_path = source_path
        self.source_split = source_split
        self.work_repo = work_repo
        self.output_repo = output_repo
        self.configuration_hash = configuration_hash
        self.prompt_config_hash = prompt_config_hash
        self.code_revision = code_revision or get_code_revision()
        self.current_stage = current_stage
        self.current_shard = current_shard
        self.completed_shards = completed_shards or []
        self.status = status
        now = time.time()
        self.timestamps = timestamps or {
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
        }
        self.error_retry_info = error_retry_info or {
            "retries": 0,
            "last_error": None,
            "errors": [],
        }
        self.final_output_revision = final_output_revision

    def update_timestamp(self):
        self.timestamps["updated_at"] = time.time()

    def mark_shard_completed(self, shard_id: Union[str, int]):
        if shard_id not in self.completed_shards:
            self.completed_shards.append(shard_id)
        self.update_timestamp()

    def set_status(self, new_status: str, error_msg: Optional[str] = None):
        self.status = new_status
        self.update_timestamp()
        if new_status in ("COMPLETED", "FAILED", "REVOKED"):
            self.timestamps["completed_at"] = time.time()
        if error_msg:
            self.error_retry_info["last_error"] = error_msg
            self.error_retry_info["errors"].append(
                {"timestamp": time.time(), "message": error_msg}
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "pipeline": self.pipeline,
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "source_path": self.source_path,
            "source_split": self.source_split,
            "work_repo": self.work_repo,
            "output_repo": self.output_repo,
            "configuration_hash": self.configuration_hash,
            "prompt_config_hash": self.prompt_config_hash,
            "code_revision": self.code_revision,
            "current_stage": self.current_stage,
            "current_shard": self.current_shard,
            "completed_shards": list(self.completed_shards),
            "status": self.status,
            "timestamps": dict(self.timestamps),
            "error_retry_info": dict(self.error_retry_info),
            "final_output_revision": self.final_output_revision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobManifest":
        return cls(
            job_id=data["job_id"],
            pipeline=data.get("pipeline", "unknown"),
            source_repo=data.get("source_repo", ""),
            source_revision=data.get("source_revision", "main"),
            source_path=data.get("source_path"),
            source_split=data.get("source_split"),
            work_repo=data.get("work_repo", ""),
            output_repo=data.get("output_repo", ""),
            configuration_hash=data.get("configuration_hash", ""),
            prompt_config_hash=data.get("prompt_config_hash", ""),
            code_revision=data.get("code_revision"),
            current_stage=data.get("current_stage", "initialized"),
            current_shard=data.get("current_shard"),
            completed_shards=data.get("completed_shards", []),
            status=data.get("status", "PENDING"),
            timestamps=data.get("timestamps"),
            error_retry_info=data.get("error_retry_info"),
            final_output_revision=data.get("final_output_revision"),
        )


class JobManifestManager:
    """Utility class for validating compatibility and constructing manifests."""

    @staticmethod
    def check_compatibility(
        existing_manifest: JobManifest,
        target_manifest: JobManifest,
        strict_code_rev: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Refuses automatic continuation when materially relevant inputs changed:
        - source_revision
        - configuration_hash
        - prompt_config_hash
        - code_revision (if strict_code_rev is True or major difference)
        """
        mismatches = []

        if existing_manifest.source_revision != target_manifest.source_revision:
            mismatches.append(
                f"source_revision changed: {existing_manifest.source_revision} -> {target_manifest.source_revision}"
            )

        if existing_manifest.configuration_hash != target_manifest.configuration_hash:
            mismatches.append(
                f"configuration_hash changed: {existing_manifest.configuration_hash} -> {target_manifest.configuration_hash}"
            )

        if existing_manifest.prompt_config_hash != target_manifest.prompt_config_hash:
            mismatches.append(
                f"prompt_config_hash changed: {existing_manifest.prompt_config_hash} -> {target_manifest.prompt_config_hash}"
            )

        if strict_code_rev and (existing_manifest.code_revision != target_manifest.code_revision):
            mismatches.append(
                f"code_revision changed: {existing_manifest.code_revision} -> {target_manifest.code_revision}"
            )

        is_compatible = len(mismatches) == 0
        return is_compatible, mismatches
