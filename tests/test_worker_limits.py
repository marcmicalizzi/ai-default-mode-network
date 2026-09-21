from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from dmn.worker_limits import WorkerLimits, WindowsJob, run_cpu_worker


class LimitsTest(unittest.TestCase):
    def test_rejects_invalid_resource_ceilings(self):
        for values in ({"max_committed_bytes": True}, {"max_committed_bytes": 2**64},
                       {"max_seconds": float("nan")}, {"max_seconds": float("inf")},
                       {"max_seconds": 0}, {"max_processes": 1}, {"max_log_bytes": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                WorkerLimits(**dict({"max_committed_bytes": 128 * 1024**2, "max_seconds": 5}, **values))

    @unittest.skipIf(os.name == "nt", "non-Windows fail-closed path")
    def test_unsupported_host_refuses_without_a_polling_ram_fallback(self):
        with self.assertRaisesRegex(NotImplementedError, "Linux containment"):
            WindowsJob(WorkerLimits(128 * 1024**2, 5))


@unittest.skipUnless(os.name == "nt", "real Windows Job Object containment")
class WindowsWorkerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.calls = 0

    def tearDown(self):
        self.temp.cleanup()

    def run_code(self, code, **limits):
        self.calls += 1
        return run_cpu_worker(sys.executable, ["-c", code], cwd=self.root,
            log=self.root / ("worker.log" if self.calls == 1 else f"worker-{self.calls}.log"), limits=WorkerLimits(**dict(
                {"max_committed_bytes": 128 * 1024**2, "max_seconds": 10}, **limits)))

    def test_success_and_log_flood_remain_bounded(self):
        result = self.run_code("print('x' * 1000000)", max_log_bytes=127)
        self.assertTrue(result["succeeded"])
        self.assertTrue(result["log_truncated"])
        self.assertGreater(result["log_bytes_seen"], 1000000)
        self.assertEqual((self.root / "worker.log").stat().st_size, 127)
        self.assertEqual(result["active_processes"], 0)
        # Windows venv launchers/console helpers also belong to the job.
        self.assertGreaterEqual(result["total_processes"], 2)
        self.assertGreater(result["os_peak_job_memory_bytes"], 0)

    def test_os_refuses_large_allocation(self):
        result = self.run_code("bytearray(256 * 1024**2)", max_committed_bytes=64 * 1024**2)
        self.assertFalse(result["succeeded"])
        self.assertNotEqual(result["returncode"], 0)
        self.assertIn("MemoryError", (self.root / "worker.log").read_text())

    def test_memory_limit_counts_parent_and_descendant_together(self):
        child = "bytearray(48 * 1024**2)"
        code = ("import subprocess,sys; retained=bytearray(48 * 1024**2); "
                f"sys.exit(subprocess.call([sys.executable,'-c',{child!r}]))")
        result = self.run_code(code, max_committed_bytes=96 * 1024**2)
        self.assertFalse(result["succeeded"])
        self.assertGreaterEqual(result["total_processes"], 3)
        self.assertIn("MemoryError", (self.root / "worker.log").read_text())
        # Either allocation fits by itself; only their combined job exceeds it.
        control = self.run_code(child, max_committed_bytes=96 * 1024**2)
        self.assertTrue(control["succeeded"])

    def test_watchdog_covers_descendant_after_workload_parent_exits(self):
        child = "import time; from pathlib import Path; Path('child-started').touch(); time.sleep(60)"
        result = self.run_code(f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}])",
                               max_seconds=2)
        self.assertTrue((self.root / "child-started").exists())
        self.assertEqual(result["outcome"], "time_limit")
        self.assertFalse(result["succeeded"])
        self.assertLess(result["elapsed_seconds"], 8)

    def test_cancellation_stops_the_tree(self):
        started = time.monotonic()
        result = run_cpu_worker(sys.executable, ["-c", "import time; time.sleep(60)"], cwd=self.root,
            log=self.root / "worker.log", limits=WorkerLimits(128 * 1024**2, 10),
            cancelled=lambda: time.monotonic() - started >= .5)
        self.assertEqual(result["outcome"], "cancelled")
        self.assertFalse(result["succeeded"])

    def test_containment_verification_failure_never_resumes_worker(self):
        with patch.object(WindowsJob, "verify_assignment", side_effect=OSError("assignment refused")):
            with self.assertRaisesRegex(OSError, "assignment refused"):
                self.run_code("from pathlib import Path; Path('should-not-exist').touch()")
        self.assertFalse((self.root / "should-not-exist").exists())

    def test_supervisor_death_terminates_worker(self):
        self.check_supervisor_death(before_resume=False)

    def test_supervisor_death_before_resume_cannot_orphan_suspended_worker(self):
        self.check_supervisor_death(before_resume=True)

    def check_supervisor_death(self, before_resume):
        # Hold a process handle before killing the supervisor, avoiding PID reuse
        # ambiguity when checking that the worker has actually exited.
        import ctypes as C
        from ctypes import wintypes as W
        api = C.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes, api.OpenProcess.restype = [W.DWORD, W.BOOL, W.DWORD], W.HANDLE
        api.WaitForSingleObject.argtypes, api.WaitForSingleObject.restype = [W.HANDLE, W.DWORD], W.DWORD
        api.CloseHandle.argtypes, api.CloseHandle.restype = [W.HANDLE], W.BOOL
        worker = "import os,time; from pathlib import Path; Path('pid').write_text(str(os.getpid())); time.sleep(60)"
        root = Path(__file__).resolve().parents[1]
        supervisor = ("import sys,time; from pathlib import Path; "
            f"sys.path.insert(0,{str(root)!r}); "
            "from dmn.worker_limits import WorkerLimits,WindowsJob,run_cpu_worker\n")
        if before_resume:
            supervisor += ("def stop_before_resume(job,process):\n"
                " Path('pid').write_text(str(process.pid))\n time.sleep(60)\n"
                "WindowsJob.verify_assignment=stop_before_resume\n")
        supervisor += (
            f"run_cpu_worker(sys.executable,['-c',{worker!r}],cwd=Path.cwd(),log=Path('worker.log'),"
            "limits=WorkerLimits(128*1024**2,60))")
        process = subprocess.Popen([sys.executable, "-c", supervisor], cwd=self.root,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        handle = None
        try:
            deadline = time.monotonic() + 8
            pid = None
            while time.monotonic() < deadline:
                try:
                    pid = int((self.root / "pid").read_text())
                    break
                except (FileNotFoundError, ValueError):
                    time.sleep(.05)
            self.assertIsNotNone(pid)
            handle = api.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            self.assertTrue(handle)
            process.kill()
            process.wait(timeout=5)
            self.assertEqual(api.WaitForSingleObject(handle, 5000), 0)  # WAIT_OBJECT_0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if handle:
                api.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
