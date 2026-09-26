import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from augmentoolkit.hf_persistence import (
    CheckpointManager,
    HFHubError,
    HFHubManager,
    IncompatibleResumeError,
    JobManifest,
    JobManifestManager,
    compute_hash,
)


class TestHFSpaceRecoveryLifecycle(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ckpt_dir = os.path.join(self.test_dir, "checkpoints")
        self.outputs_dir = os.path.join(self.test_dir, "outputs")

        self.mock_hf_mgr = MagicMock(spec=HFHubManager)
        self.mock_hf_mgr.token = "mock_token"
        self.mock_hf_mgr.source_repo = "org/private-source"
        self.mock_hf_mgr.work_repo = "org/private-work"
        self.mock_hf_mgr.output_repo = "org/private-output"

        # Virtual HF storage simulating remote HF_WORK_REPO and HF_OUTPUT_REPO
        self.remote_work_repo_files = {}
        self.remote_output_repo_files = {}

        def mock_upload_file(local_path, path_in_repo, repo_id, repo_type="dataset", commit_message=None, revision=None):
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()
            if repo_id == self.mock_hf_mgr.work_repo:
                self.remote_work_repo_files[path_in_repo] = content
            elif repo_id == self.mock_hf_mgr.output_repo:
                self.remote_output_repo_files[path_in_repo] = content
            return "commit_sha_12345"

        def mock_file_exists(repo_id, path_in_repo, repo_type="dataset", revision=None):
            if repo_id == self.mock_hf_mgr.work_repo:
                return path_in_repo in self.remote_work_repo_files
            if repo_id == self.mock_hf_mgr.output_repo:
                return path_in_repo in self.remote_output_repo_files
            return False

        def mock_download_file(path_in_repo, local_path, repo_id, repo_type="dataset", revision=None):
            if repo_id == self.mock_hf_mgr.work_repo and path_in_repo in self.remote_work_repo_files:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(self.remote_work_repo_files[path_in_repo])
                return local_path
            raise HFHubError(f"File not found in mock repo: {path_in_repo}")

        def mock_fetch_manifest(job_id, work_repo=None):
            path_in_repo = f"jobs/{job_id}/manifest.json"
            if path_in_repo in self.remote_work_repo_files:
                return json.loads(self.remote_work_repo_files[path_in_repo])
            return None

        self.mock_hf_mgr.upload_file.side_effect = mock_upload_file
        self.mock_hf_mgr.file_exists.side_effect = mock_file_exists
        self.mock_hf_mgr.download_file.side_effect = mock_download_file
        self.mock_hf_mgr.fetch_manifest.side_effect = mock_fetch_manifest

        self.ckpt_mgr = CheckpointManager(
            hf_manager=self.mock_hf_mgr,
            checkpoint_dir=self.ckpt_dir,
            outputs_dir=self.outputs_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_job_manifest_creation_and_hashing(self):
        config_a = {"chunk_size": 1000, "model": "gpt-4o"}
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id="job_001",
            pipeline="example-pipeline",
            config_dict=config_a,
            sync_remote=True,
        )

        self.assertEqual(manifest.job_id, "job_001")
        self.assertEqual(manifest.configuration_hash, compute_hash(config_a))
        self.assertIn("jobs/job_001/manifest.json", self.remote_work_repo_files)

    def test_remote_first_checkpointing_and_fail_closed(self):
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id="job_002",
            pipeline="example-pipeline",
            sync_remote=True,
        )

        ckpt_data = {"item_1": {"result": "sample output"}}
        commit_sha = self.ckpt_mgr.save_checkpoint(
            manifest=manifest,
            shard_id="shard_0",
            checkpoint_data=ckpt_data,
            sync_remote=True,
        )

        self.assertEqual(commit_sha, "commit_sha_12345")
        self.assertIn("jobs/job_002/checkpoints/shard_shard_0.json", self.remote_work_repo_files)
        self.assertIn("shard_0", manifest.completed_shards)

        # Test fail-closed behavior on upload error
        self.mock_hf_mgr.upload_file.side_effect = HFHubError("Network connection interrupted")
        with self.assertRaises(HFHubError):
            self.ckpt_mgr.save_checkpoint(
                manifest=manifest,
                shard_id="shard_1",
                checkpoint_data={"item_2": {"result": "fail test"}},
                sync_remote=True,
            )
        self.assertEqual(manifest.status, "FAILED")

    def test_incompatible_resume_detection(self):
        config_orig = {"chunk_size": 1000, "model": "gpt-4o"}
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id="job_003",
            pipeline="example-pipeline",
            source_revision="v1.0",
            config_dict=config_orig,
            sync_remote=True,
        )

        # Attempt resume with changed source revision
        with self.assertRaises(IncompatibleResumeError):
            self.ckpt_mgr.reconcile_and_resume_job(
                job_id="job_003",
                target_pipeline="example-pipeline",
                target_config_dict=config_orig,
                target_source_revision="v2.0",
            )

        # Attempt resume with changed configuration
        config_modified = {"chunk_size": 2000, "model": "gpt-4o"}
        with self.assertRaises(IncompatibleResumeError):
            self.ckpt_mgr.reconcile_and_resume_job(
                job_id="job_003",
                target_pipeline="example-pipeline",
                target_config_dict=config_modified,
                target_source_revision="v1.0",
            )

    def test_critical_recovery_lifecycle(self):
        """
        Full recovery lifecycle simulation:
        1. Start job and complete shard_0 and shard_1.
        2. Simulate process/Space death with empty local filesystem.
        3. Reconstruct state from HF_WORK_REPO.
        4. Resume job: verify shard_0 and shard_1 skipped, process only shard_2.
        5. Finalize dataset and upload to HF_OUTPUT_REPO.
        """
        job_id = "job_lifecycle_test"
        config = {"use_subset": True, "subset_size": 10}

        # Step 1: Start job & complete 2 shards remotely
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id=job_id,
            pipeline="example-pipeline",
            config_dict=config,
            sync_remote=True,
        )

        self.ckpt_mgr.save_checkpoint(
            manifest=manifest,
            shard_id="shard_0",
            checkpoint_data={"s0_k1": "v1"},
            sync_remote=True,
        )
        self.ckpt_mgr.save_checkpoint(
            manifest=manifest,
            shard_id="shard_1",
            checkpoint_data={"s1_k1": "v2"},
            sync_remote=True,
        )

        # Confirm 2 shards completed
        self.assertEqual(sorted(manifest.completed_shards), ["shard_0", "shard_1"])

        # Step 2: Simulate Space death / rebuild -> clear local filesystem
        shutil.rmtree(self.ckpt_dir)
        shutil.rmtree(self.outputs_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Step 3 & 4: Reconstruct state from remote HF_WORK_REPO and resume
        fresh_ckpt_mgr = CheckpointManager(
            hf_manager=self.mock_hf_mgr,
            checkpoint_dir=self.ckpt_dir,
            outputs_dir=self.outputs_dir,
        )

        recovered_manifest, completed_shards, restored_data = fresh_ckpt_mgr.reconcile_and_resume_job(
            job_id=job_id,
            target_pipeline="example-pipeline",
            target_config_dict=config,
            target_source_revision="main",
        )

        self.assertEqual(completed_shards, {"shard_0", "shard_1"})
        self.assertIn("s0_k1", restored_data)
        self.assertIn("s1_k1", restored_data)

        # Process remaining shard_2 without re-running shard_0 or shard_1
        shards_to_process = ["shard_0", "shard_1", "shard_2"]
        processed_shards = []

        for shard in shards_to_process:
            if shard in completed_shards:
                continue  # Skip already-completed shard!
            processed_shards.append(shard)
            fresh_ckpt_mgr.save_checkpoint(
                manifest=recovered_manifest,
                shard_id=shard,
                checkpoint_data={"s2_k1": "v3"},
                sync_remote=True,
            )

        self.assertEqual(processed_shards, ["shard_2"])
        self.assertEqual(sorted(recovered_manifest.completed_shards), ["shard_0", "shard_1", "shard_2"])

        # Step 5: Finalize dataset and verify HF_OUTPUT_REPO upload
        final_items = [{"instruction": "q1", "output": "a1"}]
        out_rev = fresh_ckpt_mgr.finalize_job(
            manifest=recovered_manifest,
            final_dataset_items=final_items,
            format_type="jsonl",
            sync_remote=True,
        )

        self.assertEqual(out_rev, "commit_sha_12345")
        self.assertIn(f"datasets/{job_id}/train.jsonl", self.remote_output_repo_files)
        self.assertEqual(recovered_manifest.status, "COMPLETED")
        self.assertEqual(recovered_manifest.final_output_revision, "commit_sha_12345")

    def test_corrupted_checkpoint_handling(self):
        job_id = "corrupted_ckpt_job"
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id=job_id,
            pipeline="example-pipeline",
            sync_remote=True,
        )
        self.ckpt_mgr.save_checkpoint(
            manifest=manifest,
            shard_id="shard_0",
            checkpoint_data={"valid": "data"},
            sync_remote=True,
        )

        # Corrupt the remote checkpoint file JSON
        ckpt_repo_path = f"jobs/{job_id}/checkpoints/shard_shard_0.json"
        self.remote_work_repo_files[ckpt_repo_path] = "INVALID_JSON{{{"

        # Reconciliation should handle bad shard JSON gracefully
        recovered_manifest, completed_shards, restored_data = self.ckpt_mgr.reconcile_and_resume_job(
            job_id=job_id,
            target_pipeline="example-pipeline",
        )
        self.assertIn("shard_0", completed_shards)

    def test_duplicate_resume_request(self):
        job_id = "duplicate_resume_job"
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id=job_id,
            pipeline="example-pipeline",
            sync_remote=True,
        )
        manifest.set_status("COMPLETED")
        self.ckpt_mgr.save_manifest_remote(manifest)

        # Re-reconcile completed job
        recovered_manifest, completed_shards, restored_data = self.ckpt_mgr.reconcile_and_resume_job(
            job_id=job_id,
            target_pipeline="example-pipeline",
        )
        self.assertEqual(recovered_manifest.job_id, job_id)


if __name__ == "__main__":
    unittest.main()
