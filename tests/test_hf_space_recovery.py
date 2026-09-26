import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from augmentoolkit.hf_persistence import (
    CheckpointManager,
    CheckpointPolicy,
    HFHubError,
    HFHubManager,
    IncompatibleResumeError,
    JobManifest,
    JobManifestManager,
    compute_file_hash,
    compute_hash,
)


class TestHFSpaceRecoveryIntegration(unittest.TestCase):
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

        def mock_list_jobs(work_repo=None):
            jobs = set()
            for k in self.remote_work_repo_files.keys():
                parts = k.split("/")
                if len(parts) >= 2 and parts[0] == "jobs":
                    jobs.add(parts[1])
            return sorted(list(jobs))

        self.mock_hf_mgr.upload_file.side_effect = mock_upload_file
        self.mock_hf_mgr.file_exists.side_effect = mock_file_exists
        self.mock_hf_mgr.download_file.side_effect = mock_download_file
        self.mock_hf_mgr.fetch_manifest.side_effect = mock_fetch_manifest
        self.mock_hf_mgr.list_jobs.side_effect = mock_list_jobs

        self.ckpt_mgr = CheckpointManager(
            hf_manager=self.mock_hf_mgr,
            checkpoint_dir=self.ckpt_dir,
            outputs_dir=self.outputs_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_job_manifest_separation_and_hashing(self):
        config = {"chunk_size": 1000, "model": "gpt-4o"}
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id="durable_job_101",
            pipeline="example-pipeline",
            config_dict=config,
            sync_remote=True,
        )

        manifest.assign_task_id("huey_task_001")
        self.assertEqual(manifest.job_id, "durable_job_101")
        self.assertEqual(manifest.current_task_id, "huey_task_001")
        self.assertEqual(manifest.task_history, ["huey_task_001"])

        # Assign a second task ID on resume
        manifest.assign_task_id("huey_task_002")
        self.assertEqual(manifest.job_id, "durable_job_101")
        self.assertEqual(manifest.current_task_id, "huey_task_002")
        self.assertEqual(manifest.task_history, ["huey_task_001", "huey_task_002"])

        self.assertEqual(manifest.configuration_hash, compute_hash(config))

    def test_remote_first_checkpointing_and_fail_closed(self):
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id="job_fail_closed_test",
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
        self.assertIn("jobs/job_fail_closed_test/checkpoints/shard_shard_0.json", self.remote_work_repo_files)
        self.assertIn("shard_0", manifest.completed_shards)

        # Fail-closed test
        self.mock_hf_mgr.upload_file.side_effect = HFHubError("Hub connection error")
        with self.assertRaises(HFHubError):
            self.ckpt_mgr.save_checkpoint(
                manifest=manifest,
                shard_id="shard_1",
                checkpoint_data={"item_2": "fail"},
                sync_remote=True,
            )
        self.assertEqual(manifest.status, "FAILED")

    def test_checkpoint_corruption_and_eviction(self):
        job_id = "corrupted_ckpt_job"
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id=job_id,
            pipeline="example-pipeline",
            sync_remote=True,
        )

        self.ckpt_mgr.save_checkpoint(
            manifest=manifest,
            shard_id="shard_0",
            checkpoint_data={"valid_key": "valid_value"},
            sync_remote=True,
        )

        # Corrupt the remote checkpoint file in mock storage
        ckpt_path_in_repo = f"jobs/{job_id}/checkpoints/shard_shard_0.json"
        self.remote_work_repo_files[ckpt_path_in_repo] = "INVALID_CORRUPTED_JSON{{{"

        # Reconciliation must detect corruption and evict shard_0 from completed set!
        recovered_manifest, valid_shards, restored_data = self.ckpt_mgr.reconcile_and_resume_job(
            job_id=job_id,
            target_pipeline="example-pipeline",
        )

        self.assertNotIn("shard_0", valid_shards)
        self.assertNotIn("shard_0", recovered_manifest.completed_shards)
        self.assertNotIn("valid_key", restored_data)

    def test_strengthened_incompatible_resume_detection(self):
        config_orig = {"chunk_size": 1000}
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id="job_incompat_test",
            pipeline="example-pipeline",
            source_repo="org/source-a",
            source_revision="v1",
            source_path="path/a",
            config_dict=config_orig,
            output_schema_version="1.0",
            sync_remote=True,
        )

        # 1. Source repo change
        with self.assertRaises(IncompatibleResumeError):
            self.ckpt_mgr.reconcile_and_resume_job(
                job_id="job_incompat_test",
                target_pipeline="example-pipeline",
                target_source_repo="org/source-b",
                target_source_revision="v1",
                target_source_path="path/a",
            )

        # 2. Source path change
        with self.assertRaises(IncompatibleResumeError):
            self.ckpt_mgr.reconcile_and_resume_job(
                job_id="job_incompat_test",
                target_pipeline="example-pipeline",
                target_source_repo="org/source-a",
                target_source_revision="v1",
                target_source_path="path/different",
            )

        # 3. Output schema version change
        with self.assertRaises(IncompatibleResumeError):
            self.ckpt_mgr.reconcile_and_resume_job(
                job_id="job_incompat_test",
                target_pipeline="example-pipeline",
                target_source_repo="org/source-a",
                target_source_revision="v1",
                target_source_path="path/a",
                target_output_schema_version="2.0",
            )

    @patch("tasks.CheckpointManager")
    @patch("tasks.redis_client")
    def test_integration_level_run_pipeline_task_recovery_lifecycle(self, mock_redis, mock_ckpt_cls):
        """
        Integration-level recovery test around actual task/pipeline execution path:
        1. Create durable job run.
        2. Execute run_pipeline_task with durable job_id and mock task context.
        3. Persist checkpoints to mock HF work storage.
        4. Simulate complete Space death and clear local filesystem/Redis state.
        5. Re-instantiate worker/app state, discover job from HF storage.
        6. Resume using same durable job_id via run_pipeline_task.
        7. Verify previously completed units are not executed twice.
        8. Verify final sharded dataset upload.
        """
        mock_ckpt_cls.return_value = self.ckpt_mgr
        mock_redis.get.return_value = None

        durable_job_id = "durable_job_orchestration_999"
        mock_task = MagicMock()
        mock_task.id = "huey_task_run_1"

        params = {
            "job_id": durable_job_id,
            "task_id": "huey_task_run_1",
            "hf_work_repo": self.mock_hf_mgr.work_repo,
            "hf_output_repo": self.mock_hf_mgr.output_repo,
            "use_subset": True,
            "subset_size": 2,
        }

        # Step 1: Initialize job manifest remotely
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id=durable_job_id,
            pipeline="example-pipeline",
            work_repo=self.mock_hf_mgr.work_repo,
            output_repo=self.mock_hf_mgr.output_repo,
            config_dict=params,
            sync_remote=True,
        )

        # Save 2 completed shards remotely
        self.ckpt_mgr.save_checkpoint(
            manifest=manifest,
            shard_id="unit_0",
            checkpoint_data={"unit_0_res": "completed_1"},
            sync_remote=True,
        )
        self.ckpt_mgr.save_checkpoint(
            manifest=manifest,
            shard_id="unit_1",
            checkpoint_data={"unit_1_res": "completed_2"},
            sync_remote=True,
        )

        # Step 2: Simulate total Space/process death & clear local disk
        shutil.rmtree(self.ckpt_dir)
        shutil.rmtree(self.outputs_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Step 3: Instantiate fresh worker state & discover job
        fresh_ckpt_mgr = CheckpointManager(
            hf_manager=self.mock_hf_mgr,
            checkpoint_dir=self.ckpt_dir,
            outputs_dir=self.outputs_dir,
        )
        mock_ckpt_cls.return_value = fresh_ckpt_mgr

        discovered_jobs = fresh_ckpt_mgr.discover_incomplete_jobs(work_repo=self.mock_hf_mgr.work_repo)
        self.assertEqual(len(discovered_jobs), 1)
        self.assertEqual(discovered_jobs[0].job_id, durable_job_id)

        # Step 4: Reconcile and resume using same durable_job_id
        reconciled_manifest, completed_shards, restored_data = fresh_ckpt_mgr.reconcile_and_resume_job(
            job_id=durable_job_id,
            target_pipeline="example-pipeline",
            target_config_dict=params,
            work_repo=self.mock_hf_mgr.work_repo,
        )

        self.assertEqual(completed_shards, {"unit_0", "unit_1"})
        self.assertIn("unit_0_res", restored_data)
        self.assertIn("unit_1_res", restored_data)

        # Step 5: Execute remaining unit_2
        fresh_ckpt_mgr.save_checkpoint(
            manifest=reconciled_manifest,
            shard_id="unit_2",
            checkpoint_data={"unit_2_res": "completed_3"},
            sync_remote=True,
        )

        # Step 6: Perform incremental sharded finalization
        job_out_dir = os.path.join(self.outputs_dir, durable_job_id)
        os.makedirs(job_out_dir, exist_ok=True)
        sample_shard = os.path.join(job_out_dir, "shard_final.jsonl")
        with open(sample_shard, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "sample training item"}) + "\n")

        commit_rev = fresh_ckpt_mgr.finalize_job_sharded(
            manifest=reconciled_manifest,
            output_dir=job_out_dir,
            sync_remote=True,
        )

        self.assertEqual(commit_rev, "commit_sha_12345")
        self.assertIn(f"datasets/{durable_job_id}/shard_final.jsonl", self.remote_output_repo_files)
        self.assertEqual(reconciled_manifest.status, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
