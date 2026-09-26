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
    canonical_json = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_file_hash(filepath: str) -> str:
    """Computes SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_dir_hash(dir_path: str) -> str:
    """Computes hash of all files in a directory deterministically."""
    if not dir_path or not os.path.exists(dir_path):
        return compute_hash(f"EMPTY_OR_NONEXISTENT:{dir_path}")

    hashes = []
    for root, _, files in sorted(os.walk(dir_path)):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                rel_path = os.path.relpath(fpath, dir_path)
                file_h = compute_file_hash(fpath)
                hashes.append(f"{rel_path}:{file_h}")
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
    """
    Encapsulates the state and metadata of a durable dataset generation job.
    Maintains strict separation between durable job_id and ephemeral task_id (Huey execution).
    """

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
        output_schema_version: str = "1.0",
        current_stage: str = "initialized",
        current_shard: Optional[Union[str, int]] = None,
        completed_shards: Optional[List[Union[str, int]]] = None,
        checkpoints_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        output_shards: Optional[List[Dict[str, Any]]] = None,
        status: str = "PENDING",
        current_task_id: Optional[str] = None,
        task_history: Optional[List[str]] = None,
        timestamps: Optional[Dict[str, Any]] = None,
        error_retry_info: Optional[Dict[str, Any]] = None,
        final_output_revision: Optional[str] = None,
    ):
        self.job_id = job_id  # Durable identifier across restarts/resumes
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
        self.output_schema_version = output_schema_version

        self.current_stage = current_stage
        self.current_shard = current_shard
        self.completed_shards = completed_shards or []
        self.checkpoints_metadata = checkpoints_metadata or {}
        self.output_shards = output_shards or []

        self.status = status
        self.current_task_id = current_task_id  # Ephemeral Huey task ID
        self.task_history = task_history or []
        if current_task_id and current_task_id not in self.task_history:
            self.task_history.append(current_task_id)

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

    def assign_task_id(self, task_id: str):
        """Associates a new ephemeral execution task_id with this durable job."""
        self.current_task_id = task_id
        if task_id not in self.task_history:
            self.task_history.append(task_id)
        self.update_timestamp()

    def update_timestamp(self):
        self.timestamps["updated_at"] = time.time()

    def mark_shard_completed(
        self,
        shard_id: Union[str, int],
        checksum: Optional[str] = None,
        size: Optional[int] = None,
        record_count: Optional[int] = None,
    ):
        shard_str = str(shard_id)
        if shard_str not in [str(s) for s in self.completed_shards]:
            self.completed_shards.append(shard_id)

        self.checkpoints_metadata[shard_str] = {
            "checksum": checksum,
            "size": size,
            "record_count": record_count,
            "updated_at": time.time(),
        }
        self.update_timestamp()

    def record_output_shard(
        self,
        filename: str,
        record_count: int,
        sha256: str,
        commit_sha: str,
        path_in_repo: str,
    ):
        shard_record = {
            "filename": filename,
            "record_count": record_count,
            "sha256": sha256,
            "commit_sha": commit_sha,
            "path_in_repo": path_in_repo,
            "uploaded_at": time.time(),
        }
        self.output_shards = [s for s in self.output_shards if s["filename"] != filename]
        self.output_shards.append(shard_record)
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
            "current_task_id": self.current_task_id,
            "task_history": list(self.task_history),
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
            "output_schema_version": self.output_schema_version,
            "current_stage": self.current_stage,
            "current_shard": self.current_shard,
            "completed_shards": list(self.completed_shards),
            "checkpoints_metadata": dict(self.checkpoints_metadata),
            "output_shards": list(self.output_shards),
            "status": self.status,
            "timestamps": dict(self.timestamps),
            "error_retry_info": dict(self.error_retry_info),
            "final_output_revision": self.final_output_revision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobManifest":
        return cls(
            job_id=data["job_id"],
            current_task_id=data.get("current_task_id"),
            task_history=data.get("task_history", []),
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
            output_schema_version=data.get("output_schema_version", "1.0"),
            current_stage=data.get("current_stage", "initialized"),
            current_shard=data.get("current_shard"),
            completed_shards=data.get("completed_shards", []),
            checkpoints_metadata=data.get("checkpoints_metadata", {}),
            output_shards=data.get("output_shards", []),
            status=data.get("status", "PENDING"),
            timestamps=data.get("timestamps"),
            error_retry_info=data.get("error_retry_info"),
            final_output_revision=data.get("final_output_revision"),
        )


class JobManifestManager:
    """Utility class for validating compatibility and constructing manifests."""

    EXCLUDED_RUNTIME_KEYS = {
        "task_id",
        "job_id",
        "current_task_id",
        "task_history",
        "restored_shards",
        "completed_shards",
        "output_dir",
        "input_dir",
        "chunking_output_dir",
        "no_flatten",
        "hf_source_repo",
        "hf_work_repo",
        "hf_output_repo",
        "timestamps",
        "status",
        "error_retry_info",
    }

    @classmethod
    def canonical_generation_spec(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Strips ephemeral runtime parameters to yield a canonical immutable generation specification."""
        if not isinstance(config_dict, dict):
            return {}
        spec = {}
        for k, v in config_dict.items():
            if k in cls.EXCLUDED_RUNTIME_KEYS:
                continue
            if isinstance(v, dict):
                spec[k] = cls.canonical_generation_spec(v)
            else:
                spec[k] = v
        return spec

    @classmethod
    def canonical_generation_spec_hash(cls, config_dict: Dict[str, Any]) -> str:
        """Computes deterministic SHA256 hash of canonical generation spec."""
        spec = cls.canonical_generation_spec(config_dict)
        return compute_hash(spec)

    @classmethod
    def compute_live_prompt_hash(
        cls,
        prompt_folder: Optional[str] = None,
        default_prompt_folder: Optional[str] = None,
    ) -> str:
        """Computes SHA256 hash from live prompt files on disk."""
        dir_hashes = []
        for pdir in (prompt_folder, default_prompt_folder):
            if pdir and os.path.exists(pdir):
                dir_hashes.append(compute_dir_hash(pdir))
        if not dir_hashes:
            return compute_hash("DEFAULT_PROMPTS_SPEC")
        return compute_hash(dir_hashes)

    @staticmethod
    def check_compatibility(
        existing_manifest: JobManifest,
        target_manifest: JobManifest,
        strict_code_rev: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Refuses automatic continuation when materially relevant inputs changed:
        - source_repo / source_revision / source_path / source_split
        - merged canonical generation config hash
        - generation prompt hash
        - output schema version
        - code revision (by default)
        """
        mismatches = []

        if existing_manifest.source_repo != target_manifest.source_repo:
            mismatches.append(f"source_repo changed: '{existing_manifest.source_repo}' -> '{target_manifest.source_repo}'")

        if existing_manifest.source_revision != target_manifest.source_revision:
            mismatches.append(f"source_revision changed: '{existing_manifest.source_revision}' -> '{target_manifest.source_revision}'")

        if existing_manifest.source_path != target_manifest.source_path:
            mismatches.append(f"source_path changed: '{existing_manifest.source_path}' -> '{target_manifest.source_path}'")

        if existing_manifest.source_split != target_manifest.source_split:
            mismatches.append(f"source_split changed: '{existing_manifest.source_split}' -> '{target_manifest.source_split}'")

        if existing_manifest.configuration_hash != target_manifest.configuration_hash:
            mismatches.append(f"configuration_hash changed: '{existing_manifest.configuration_hash}' -> '{target_manifest.configuration_hash}'")

        if existing_manifest.prompt_config_hash and target_manifest.prompt_config_hash:
            if existing_manifest.prompt_config_hash != target_manifest.prompt_config_hash:
                mismatches.append(f"prompt_config_hash changed: '{existing_manifest.prompt_config_hash}' -> '{target_manifest.prompt_config_hash}'")

        if existing_manifest.output_schema_version != target_manifest.output_schema_version:
            mismatches.append(f"output_schema_version changed: '{existing_manifest.output_schema_version}' -> '{target_manifest.output_schema_version}'")

        if strict_code_rev and (existing_manifest.code_revision != target_manifest.code_revision):
            mismatches.append(f"code_revision changed: '{existing_manifest.code_revision}' -> '{target_manifest.code_revision}'")

        is_compatible = len(mismatches) == 0
        return is_compatible, mismatches
