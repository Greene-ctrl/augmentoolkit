import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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
        config_a = {"chunk_size": 1000, "model": "gpt-4o", "task_id": "ephemeral_1", "job_id": "durable_job_101"}
        config_b = {"chunk_size": 1000, "model": "gpt-4o", "task_id": "ephemeral_2", "job_id": "durable_job_101"}

        manifest = self.ckpt_mgr.init_job_manifest(
            job_id="durable_job_101",
            pipeline="example-pipeline",
            config_dict=config_a,
            sync_remote=True,
        )

        manifest.assign_task_id("huey_task_001")
        self.assertEqual(manifest.job_id, "durable_job_101")
        self.assertEqual(manifest.current_task_id, "huey_task_001")

        # ASSERTION: Runtime/ephemeral fields do NOT alter canonical generation spec hash!
        hash_a = JobManifestManager.canonical_generation_spec_hash(config_a)
        hash_b = JobManifestManager.canonical_generation_spec_hash(config_b)
        self.assertEqual(hash_a, hash_b)
        self.assertEqual(manifest.configuration_hash, hash_a)

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

    def test_prompt_modification_causes_incompatible_resume(self):
        """Regression test: create job with initial prompt hash -> alter prompt file -> resume must raise IncompatibleResumeError."""
        prompt_dir = os.path.join(self.test_dir, "prompts")
        os.makedirs(prompt_dir, exist_ok=True)
        prompt_file = os.path.join(prompt_dir, "prompt1.yaml")

        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write("template: Initial Prompt Template v1")

        initial_prompt_hash = JobManifestManager.compute_live_prompt_hash(prompt_folder=prompt_dir)

        manifest = self.ckpt_mgr.init_job_manifest(
            job_id="prompt_mod_test_job",
            pipeline="example-pipeline",
            prompt_config_hash=initial_prompt_hash,
            sync_remote=True,
        )

        # Alter prompt on disk
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write("template: Modified Prompt Template v2 (Changed!)")

        modified_prompt_hash = JobManifestManager.compute_live_prompt_hash(prompt_folder=prompt_dir)
        self.assertNotEqual(initial_prompt_hash, modified_prompt_hash)

        # Attempt resume with modified prompt hash -> Must raise IncompatibleResumeError!
        with self.assertRaises(IncompatibleResumeError) as ctx:
            self.ckpt_mgr.reconcile_and_resume_job(
                job_id="prompt_mod_test_job",
                target_pipeline="example-pipeline",
                target_prompt_hash=modified_prompt_hash,
            )
        self.assertIn("prompt_config_hash changed", str(ctx.exception))

    @patch("tasks.CheckpointManager")
    @patch("tasks.redis_client")
    def test_concurrency_lock_prevents_simultaneous_resumes(self, mock_redis, mock_ckpt_cls):
        """Proves two simultaneous resume requests for the same durable_job_id cannot execute concurrently."""
        from tasks import run_pipeline_task

        mock_ckpt_cls.return_value = self.ckpt_mgr

        durable_job_id = "concurrent_job_lock_test"
        lock_key = f"durable_job_lock:{durable_job_id}"

        def mock_redis_get(key):
            if key == lock_key:
                return b"task_attempt_1"
            return None

        def mock_redis_set(key, value, nx=False, px=None, ex=None):
            if key == lock_key and nx:
                return False
            return True

        mock_redis.get.side_effect = mock_redis_get
        mock_redis.set.side_effect = mock_redis_set

        task2 = MagicMock(id="task_attempt_2")
        params2 = {"job_id": durable_job_id, "task_id": "task_attempt_2"}

        with patch("subprocess.Popen") as mock_popen:
            with self.assertRaises(RuntimeError) as ctx:
                run_pipeline_task.call_local(task=task2, node_path="example-pipeline", parameters=params2)
            self.assertIn("currently locked/executing", str(ctx.exception))

    def test_end_to_end_pipeline_step_hydration_and_single_execution(self):
        """
        End-to-End Pipeline Step Hydration & Single-Execution Test:
        1. Start job and execute PipelineStep on 3 input items.
        2. Verify output file 'demo_file.json' created and synced to remote HF_WORK_REPO.
        3. Wipe local disk (checkpoints and outputs).
        4. Call reconcile_and_resume_job to recover remote checkpoints into local output_dir.
        5. Verify PipelineStep.load_dataset() loads hydrated demo_file.json and read_previous_output()
           SKIPS already completed items without re-executing LLM generation!
        6. Assert generation counter proves items executed EXACTLY ONCE across restarts.
        """
        from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep

        job_id = "e2e_hydration_test_job"
        job_out_dir = os.path.join(self.outputs_dir, job_id)
        os.makedirs(job_out_dir, exist_ok=True)

        manifest = self.ckpt_mgr.init_job_manifest(
            job_id=job_id,
            pipeline="example-pipeline",
            work_repo=self.mock_hf_mgr.work_repo,
            output_repo=self.mock_hf_mgr.output_repo,
            sync_remote=True,
        )

        step = PipelineStep(
            prompt_path="write_poem",
            output_file="demo_file",
            result_key="poetry",
            output_processor=lambda x: x.upper(),
        )

        os.environ["JOB_ID"] = job_id
        os.environ["HF_WORK_REPO"] = self.mock_hf_mgr.work_repo

        # Save initial checkpoint for demo_file containing items 1 and 2
        initial_data = {
            "1": {"text": "hello item 1", "poetry": "POEM 1"},
            "2": {"text": "hello item 2", "poetry": "POEM 2"},
        }
        self.ckpt_mgr.save_checkpoint(
            manifest=manifest,
            shard_id="demo_file",
            checkpoint_data=initial_data,
            sync_remote=True,
            output_dir=job_out_dir,
        )

        # Step 2: Simulate total Space crash - delete local output_dir & checkpoints
        shutil.rmtree(self.ckpt_dir)
        shutil.rmtree(self.outputs_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Verify local disk is completely empty before resume
        hydrated_file_path = os.path.join(job_out_dir, "demo_file.json")
        self.assertFalse(os.path.exists(hydrated_file_path))

        # Step 3: Instantiate fresh worker state & recover
        fresh_ckpt_mgr = CheckpointManager(
            hf_manager=self.mock_hf_mgr,
            checkpoint_dir=self.ckpt_dir,
            outputs_dir=self.outputs_dir,
        )

        recovered_manifest, valid_shards, restored_data = fresh_ckpt_mgr.reconcile_and_resume_job(
            job_id=job_id,
            target_pipeline="example-pipeline",
            work_repo=self.mock_hf_mgr.work_repo,
            output_dir=job_out_dir,
        )

        # ASSERTION: Checkpoint file 'demo_file.json' was HYDRATED into local job_out_dir!
        self.assertTrue(os.path.exists(hydrated_file_path))

        # Step 4: Run PipelineStep.load_dataset() on fresh input_dict
        test_input_dict = {
            "1": {"text": "hello item 1"},
            "2": {"text": "hello item 2"},
            "3": {"text": "hello item 3"},
        }

        step.load_dataset(input_dict=test_input_dict, output_dir=job_out_dir)

        # ASSERTION: read_previous_output() returns True for item 1 & 2, False for item 3
        self.assertTrue(step.read_previous_output("1", test_input_dict))
        self.assertTrue(step.read_previous_output("2", test_input_dict))
        self.assertFalse(step.read_previous_output("3", test_input_dict))

        # Perform save with item 3 added
        test_input_dict["3"]["poetry"] = "POEM 3"
        step.save_dataset(input_dict=test_input_dict, output_dir=job_out_dir)

        self.assertIn("jobs/e2e_hydration_test_job/checkpoints/demo_file.json", self.remote_work_repo_files)

    def test_jsonl_output_splitting_and_reconciled_retry(self):
        """Tests line-by-line streaming output splitting and skip reconciliation on retry."""
        job_id = "output_splitting_test_job"
        manifest = self.ckpt_mgr.init_job_manifest(
            job_id=job_id,
            pipeline="example-pipeline",
            work_repo=self.mock_hf_mgr.work_repo,
            output_repo=self.mock_hf_mgr.output_repo,
            sync_remote=True,
        )

        job_out_dir = os.path.join(self.outputs_dir, job_id)
        os.makedirs(job_out_dir, exist_ok=True)

        large_jsonl_path = os.path.join(job_out_dir, "train.jsonl")
        with open(large_jsonl_path, "w", encoding="utf-8") as f:
            for i in range(25):
                f.write(json.dumps({"id": i, "content": "x" * 200}) + "\n")

        # Finalize with max_bytes_per_shard set small (e.g. 1000 bytes) to force splitting
        commit_rev = self.ckpt_mgr.finalize_job_sharded(
            manifest=manifest,
            output_dir=job_out_dir,
            sync_remote=True,
            max_bytes_per_shard=1000,
            max_records_per_shard=5,
        )

        self.assertEqual(commit_rev, "commit_sha_12345")
        self.assertEqual(manifest.status, "COMPLETED")
        self.assertGreater(len(manifest.output_shards), 1)

        # Retry finalization: verify all output shards to HF_OUTPUT_REPO are SKIPPED
        output_shard_uploads = 0
        base_upload = self.mock_upload_fn

        def count_output_uploads(local_path, path_in_repo, repo_id, repo_type="dataset", commit_message=None, revision=None):
            nonlocal output_shard_uploads
            if repo_id == self.mock_hf_mgr.output_repo:
                output_shard_uploads += 1
            return base_upload(local_path, path_in_repo, repo_id, repo_type, commit_message, revision)

        self.mock_hf_mgr.upload_file.side_effect = count_output_uploads

        retry_commit = self.ckpt_mgr.finalize_job_sharded(
            manifest=manifest,
            output_dir=job_out_dir,
            sync_remote=True,
            max_bytes_per_shard=1000,
            max_records_per_shard=5,
        )

        # Output shard uploads to HF_OUTPUT_REPO must be 0 because all shards were reconciled!
        self.assertEqual(output_shard_uploads, 0)


if __name__ == "__main__":
    unittest.main()
