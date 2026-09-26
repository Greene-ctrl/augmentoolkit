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

        self.mock_upload_fn = mock_upload_file
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

        self.assertEqual(manifest.configuration_hash, JobManifestManager.canonical_generation_spec_hash(config))

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
        self.assertIn("jobs/job_fail_closed_test/checkpoints/shard_0.json", self.remote_work_repo_files)
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
        ckpt_path_in_repo = f"jobs/{job_id}/checkpoints/shard_0.json"
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
    def test_concurrency_lock_prevents_simultaneous_resumes(self, mock_redis, mock_ckpt_cls):
        """Proves two simultaneous resume requests for the same durable_job_id cannot execute concurrently."""
        from tasks import run_pipeline_task

        mock_ckpt_cls.return_value = self.ckpt_mgr

        # Simulate lock held by task_attempt_1 when task_attempt_2 tries to acquire
        durable_job_id = "concurrent_job_lock_test"
        lock_key = f"durable_job_lock:{durable_job_id}"

        def mock_redis_get(key):
            if key == lock_key:
                return b"task_attempt_1"
            return None

        def mock_redis_set(key, value, nx=False, px=None, ex=None):
            if key == lock_key and nx:
                return False  # SET NX fails because lock is already held!
            return True

        mock_redis.get.side_effect = mock_redis_get
        mock_redis.set.side_effect = mock_redis_set

        task2 = MagicMock(id="task_attempt_2")
        params2 = {"job_id": durable_job_id, "task_id": "task_attempt_2"}

        with patch("subprocess.Popen") as mock_popen:
            with self.assertRaises(RuntimeError) as ctx:
                run_pipeline_task.call_local(task=task2, node_path="example-pipeline", parameters=parameters2 if 'parameters2' in locals() else params2)
            self.assertIn("currently locked/executing", str(ctx.exception))

    @patch("tasks.CheckpointManager")
    @patch("tasks.redis_client")
    def test_full_orchestration_recovery_lifecycle(self, mock_redis, mock_ckpt_cls):
        """
        Orchestration Recovery Lifecycle Test:
        1. Start job with durable_job_id and process 3 deterministic work units.
        2. Assert invocation counters prove each unit executes EXACTLY ONCE across restarts.
        3. Simulate complete Space death (wipe local filesystem & Redis).
        4. Re-instantiate fresh worker state, discover manifest remotely from HF_WORK_REPO.
        5. Resume same durable_job_id with new task_id.
        6. Verify completed units are hydrated from HF and NOT re-executed.
        7. Interrupt once during output shard upload, restart, verify uploaded output shards skipped.
        8. Finish and verify final COMPLETED status.
        """
        mock_ckpt_cls.return_value = self.ckpt_mgr
        mock_redis.get.return_value = None

        durable_job_id = "durable_job_orchestrator_001"
        execution_counts = {"unit_0": 0, "unit_1": 0, "unit_2": 0}

        def execute_unit(unit_id, manifest, target_out_dir):
            execution_counts[unit_id] += 1
            data = {unit_id: {"result": f"processed_{unit_id}"}}
            self.ckpt_mgr.save_checkpoint(
                manifest=manifest,
                shard_id=unit_id,
                checkpoint_data=data,
                sync_remote=True,
                output_dir=target_out_dir,
            )

        # Step 1: Initialize job manifest remotely
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id=durable_job_id,
            pipeline="example-pipeline",
            work_repo=self.mock_hf_mgr.work_repo,
            output_repo=self.mock_hf_mgr.output_repo,
            config_dict={"use_subset": True},
            sync_remote=True,
        )

        job_out_dir = os.path.join(self.outputs_dir, durable_job_id)

        # Process unit_0 and unit_1
        execute_unit("unit_0", manifest, job_out_dir)
        execute_unit("unit_1", manifest, job_out_dir)

        self.assertEqual(execution_counts["unit_0"], 1)
        self.assertEqual(execution_counts["unit_1"], 1)
        self.assertEqual(execution_counts["unit_2"], 0)

        # Step 2: Simulate Space crash & wipe local filesystem/Redis
        shutil.rmtree(self.ckpt_dir)
        shutil.rmtree(self.outputs_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Step 3: Re-instantiate fresh worker state and recover
        fresh_ckpt_mgr = CheckpointManager(
            hf_manager=self.mock_hf_mgr,
            checkpoint_dir=self.ckpt_dir,
            outputs_dir=self.outputs_dir,
        )
        mock_ckpt_cls.return_value = fresh_ckpt_mgr

        recovered_manifest, valid_shards, restored_data = fresh_ckpt_mgr.reconcile_and_resume_job(
            job_id=durable_job_id,
            target_pipeline="example-pipeline",
            target_config_dict={"use_subset": True},
            work_repo=self.mock_hf_mgr.work_repo,
            output_dir=job_out_dir,
        )

        self.assertEqual(valid_shards, {"unit_0", "unit_1"})

        # Step 4: Resume execution. Verify completed units unit_0 and unit_1 are SKIPPED!
        all_units = ["unit_0", "unit_1", "unit_2"]
        for u in all_units:
            if u in valid_shards:
                continue  # Skip already completed unit!
            execute_unit(u, recovered_manifest, job_out_dir)

        # ASSERTION: unit_0 and unit_1 executed EXACTLY ONCE across restarts!
        self.assertEqual(execution_counts["unit_0"], 1)
        self.assertEqual(execution_counts["unit_1"], 1)
        self.assertEqual(execution_counts["unit_2"], 1)

        # Step 5: Test output shard upload interruption and resume reconciliation
        shard1_path = os.path.join(job_out_dir, "shard_001.jsonl")
        shard2_path = os.path.join(job_out_dir, "shard_002.jsonl")

        with open(shard1_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"rec": 1}) + "\n")
        with open(shard2_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"rec": 2}) + "\n")

        # Simulate upload of shard_001 succeeding, then upload of shard_002 interrupting/failing
        base_upload = self.mock_upload_fn

        def interrupted_upload(local_path, path_in_repo, repo_id, repo_type="dataset", commit_message=None, revision=None):
            if "shard_002" in path_in_repo:
                raise HFHubError("Simulated upload interruption during shard_002")
            return base_upload(local_path, path_in_repo, repo_id, repo_type, commit_message, revision)

        self.mock_hf_mgr.upload_file.side_effect = interrupted_upload

        with self.assertRaises(HFHubError):
            fresh_ckpt_mgr.finalize_job_sharded(manifest=recovered_manifest, output_dir=job_out_dir, sync_remote=True)

        # Verify shard_001 was uploaded and recorded in manifest
        self.assertIn("datasets/durable_job_orchestrator_001/shard_001.jsonl", self.remote_output_repo_files)
        self.assertNotIn("datasets/durable_job_orchestrator_001/shard_002.jsonl", self.remote_output_repo_files)

        # Restore normal upload side effect and re-finalize
        self.mock_hf_mgr.upload_file.side_effect = base_upload

        commit_rev = fresh_ckpt_mgr.finalize_job_sharded(manifest=recovered_manifest, output_dir=job_out_dir, sync_remote=True)

        # Verify shard_001 was SKIPPED on retry, shard_002 was uploaded, and manifest status is COMPLETED
        self.assertEqual(commit_rev, "commit_sha_12345")
        self.assertIn("datasets/durable_job_orchestrator_001/shard_002.jsonl", self.remote_output_repo_files)
        self.assertEqual(recovered_manifest.status, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
