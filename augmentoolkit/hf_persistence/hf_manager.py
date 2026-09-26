import os
import shutil
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, HFValidationError

logger = logging.getLogger(__name__)


class HFHubError(Exception):
    """Raised when a Hugging Face Hub operation fails or fails verification."""
    pass


class HFHubManager:
    """Manages interactions with Hugging Face Hub dataset repositories with fail-closed semantics."""

    def __init__(
        self,
        token: Optional[str] = None,
        source_repo: Optional[str] = None,
        work_repo: Optional[str] = None,
        output_repo: Optional[str] = None,
    ):
        self.token = token or os.environ.get("HF_TOKEN")
        self.source_repo = source_repo or os.environ.get("HF_SOURCE_REPO", "")
        self.work_repo = work_repo or os.environ.get("HF_WORK_REPO", "")
        self.output_repo = output_repo or os.environ.get("HF_OUTPUT_REPO", "")

        self.api = HfApi(token=self.token)

    def check_auth(self) -> Dict[str, Any]:
        """Verifies Hugging Face authentication token."""
        if not self.token:
            return {
                "authenticated": False,
                "user": None,
                "error": "HF_TOKEN environment variable/secret is not set.",
            }
        try:
            user_info = self.api.whoami(token=self.token)
            username = user_info.get("name") or user_info.get("fullname") or "authenticated_user"
            return {"authenticated": True, "user": username, "error": None}
        except Exception as e:
            logger.error(f"HF Authentication check failed: {e}")
            return {"authenticated": False, "user": None, "error": str(e)}

    def check_repo_access(
        self, repo_id: str, repo_type: str = "dataset"
    ) -> Dict[str, Any]:
        """Verifies repository existence and access permissions."""
        if not repo_id:
            return {"accessible": False, "repo_id": repo_id, "error": "Repo ID is empty."}
        try:
            repo_info = self.api.repo_info(
                repo_id=repo_id, repo_type=repo_type, token=self.token
            )
            return {
                "accessible": True,
                "repo_id": repo_id,
                "private": getattr(repo_info, "private", True),
                "error": None,
            }
        except Exception as e:
            logger.error(f"Repo access check failed for {repo_id}: {e}")
            return {"accessible": False, "repo_id": repo_id, "error": str(e)}

    def check_all_health(self) -> Dict[str, Any]:
        """Checks authentication and accessibility of source, work, and output repos."""
        auth_status = self.check_auth()
        source_status = self.check_repo_access(self.source_repo) if self.source_repo else {"accessible": False, "repo_id": "", "error": "Not configured"}
        work_status = self.check_repo_access(self.work_repo) if self.work_repo else {"accessible": False, "repo_id": "", "error": "Not configured"}
        output_status = self.check_repo_access(self.output_repo) if self.output_repo else {"accessible": False, "repo_id": "", "error": "Not configured"}

        is_healthy = (
            auth_status["authenticated"]
            and (not self.source_repo or source_status["accessible"])
            and (not self.work_repo or work_status["accessible"])
            and (not self.output_repo or output_status["accessible"])
        )

        return {
            "healthy": is_healthy,
            "auth": auth_status,
            "source_repo": source_status,
            "work_repo": work_status,
            "output_repo": output_status,
        }

    def upload_file(
        self,
        local_path: str,
        path_in_repo: str,
        repo_id: Optional[str] = None,
        repo_type: str = "dataset",
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Uploads a single file to Hugging Face Hub.
        Verifies upload success and returns commit SHA.
        Fails closed on any error.
        """
        target_repo = repo_id or self.work_repo
        if not target_repo:
            raise HFHubError("No repository specified for upload.")

        if not os.path.exists(local_path):
            raise HFHubError(f"Local file does not exist: {local_path}")

        commit_msg = commit_message or f"Upload {path_in_repo}"
        try:
            commit_info = self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=target_repo,
                repo_type=repo_type,
                token=self.token,
                revision=revision,
                commit_message=commit_msg,
            )
            commit_sha = getattr(commit_info, "oid", None) or getattr(commit_info, "commit_id", "uploaded")
            logger.info(f"Successfully uploaded {local_path} to {target_repo}/{path_in_repo} (Commit: {commit_sha})")

            # Verification: ensure file exists in repo
            if not self.file_exists(target_repo, path_in_repo, repo_type=repo_type, revision=revision):
                raise HFHubError(f"Verification failed: Uploaded file {path_in_repo} not found in {target_repo}")

            return str(commit_sha)
        except Exception as e:
            msg = f"Failed closed: Error uploading {local_path} to {target_repo}/{path_in_repo}: {e}"
            logger.error(msg)
            raise HFHubError(msg) from e

    def upload_folder(
        self,
        local_dir: str,
        path_in_repo: str,
        repo_id: Optional[str] = None,
        repo_type: str = "dataset",
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Uploads an entire directory to Hugging Face Hub.
        Verifies upload success and returns commit SHA.
        Fails closed on error.
        """
        target_repo = repo_id or self.work_repo
        if not target_repo:
            raise HFHubError("No repository specified for folder upload.")

        if not os.path.exists(local_dir):
            raise HFHubError(f"Local directory does not exist: {local_dir}")

        commit_msg = commit_message or f"Upload folder {path_in_repo}"
        try:
            commit_info = self.api.upload_folder(
                folder_path=local_dir,
                path_in_repo=path_in_repo,
                repo_id=target_repo,
                repo_type=repo_type,
                token=self.token,
                revision=revision,
                commit_message=commit_msg,
            )
            commit_sha = getattr(commit_info, "oid", None) or getattr(commit_info, "commit_id", "uploaded")
            logger.info(f"Successfully uploaded folder {local_dir} to {target_repo}/{path_in_repo} (Commit: {commit_sha})")
            return str(commit_sha)
        except Exception as e:
            msg = f"Failed closed: Error uploading folder {local_dir} to {target_repo}/{path_in_repo}: {e}"
            logger.error(msg)
            raise HFHubError(msg) from e

    def download_file(
        self,
        path_in_repo: str,
        local_path: str,
        repo_id: Optional[str] = None,
        repo_type: str = "dataset",
        revision: Optional[str] = None,
    ) -> str:
        """Downloads a single file from HF Hub to local_path."""
        target_repo = repo_id or self.source_repo
        if not target_repo:
            raise HFHubError("No repository specified for download.")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            downloaded_file = hf_hub_download(
                repo_id=target_repo,
                filename=path_in_repo,
                repo_type=repo_type,
                token=self.token,
                revision=revision,
            )
            shutil.copy(downloaded_file, local_path)
            logger.info(f"Downloaded {target_repo}/{path_in_repo} to {local_path}")
            return local_path
        except Exception as e:
            msg = f"Error downloading {path_in_repo} from {target_repo}: {e}"
            logger.error(msg)
            raise HFHubError(msg) from e

    def download_folder(
        self,
        path_in_repo: str,
        local_dir: str,
        repo_id: Optional[str] = None,
        repo_type: str = "dataset",
        revision: Optional[str] = None,
    ) -> str:
        """Downloads a subfolder from HF Hub to local_dir."""
        target_repo = repo_id or self.source_repo
        if not target_repo:
            raise HFHubError("No repository specified for folder download.")

        os.makedirs(local_dir, exist_ok=True)
        try:
            pattern = f"{path_in_repo}/*" if path_in_repo and path_in_repo != "." else None
            snapshot_dir = snapshot_download(
                repo_id=target_repo,
                repo_type=repo_type,
                token=self.token,
                revision=revision,
                allow_patterns=pattern,
            )
            source_sub = os.path.join(snapshot_dir, path_in_repo) if path_in_repo and path_in_repo != "." else snapshot_dir
            if os.path.exists(source_sub):
                for item in os.listdir(source_sub):
                    s = os.path.join(source_sub, item)
                    d = os.path.join(local_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
            logger.info(f"Downloaded folder {target_repo}/{path_in_repo} to {local_dir}")
            return local_dir
        except Exception as e:
            msg = f"Error downloading folder {path_in_repo} from {target_repo}: {e}"
            logger.error(msg)
            raise HFHubError(msg) from e

    def file_exists(
        self,
        repo_id: str,
        path_in_repo: str,
        repo_type: str = "dataset",
        revision: Optional[str] = None,
    ) -> bool:
        """Checks if a file exists in the repository."""
        try:
            return self.api.file_exists(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type=repo_type,
                token=self.token,
                revision=revision,
            )
        except Exception:
            return False

    def list_jobs(self, work_repo: Optional[str] = None) -> List[str]:
        """Lists all job IDs present in the work repository under jobs/."""
        target_repo = work_repo or self.work_repo
        if not target_repo:
            return []

        try:
            files = self.api.list_repo_files(
                repo_id=target_repo, repo_type="dataset", token=self.token
            )
            job_ids = set()
            for f in files:
                parts = f.split("/")
                if len(parts) >= 2 and parts[0] == "jobs":
                    job_ids.add(parts[1])
            return sorted(list(job_ids))
        except Exception as e:
            logger.error(f"Error listing jobs from {target_repo}: {e}")
            return []

    def fetch_manifest(
        self, job_id: str, work_repo: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetches and parses job manifest JSON from work repository."""
        import json
        import tempfile

        target_repo = work_repo or self.work_repo
        manifest_path = f"jobs/{job_id}/manifest.json"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.download_file(
                path_in_repo=manifest_path,
                local_path=tmp_path,
                repo_id=target_repo,
                repo_type="dataset",
            )
            with open(tmp_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not fetch manifest for job {job_id} from {target_repo}: {e}")
            return None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
