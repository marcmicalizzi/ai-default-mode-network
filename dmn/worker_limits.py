"""Windows process-tree limits for trusted, offline research workers.

This is not a code sandbox or a production sleep executor. In particular it
does not enforce disk or total-process VRAM quotas, or authorize a recipe,
dataset or weight change. CUDA is hidden in the default CPU entrypoint; the
separate GPU research entrypoint requires explicit opt-in.
Unsupported hosts fail before launching anything; there is no polling-only RAM
fallback. The inference environment needs no additional dependencies.
"""
from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
import subprocess
import threading
import time


@dataclasses.dataclass(frozen=True)
class WorkerLimits:
    max_committed_bytes: int
    max_seconds: float
    max_processes: int = 16
    max_log_bytes: int = 1024 * 1024

    def __post_init__(self):
        if type(self.max_committed_bytes) is not int or not 32 * 1024**2 <= self.max_committed_bytes <= 2**63 - 1:
            raise ValueError("worker committed-memory ceiling must be at least 32 MiB")
        if (type(self.max_seconds) not in (int, float) or not math.isfinite(self.max_seconds) or
                not 0 < self.max_seconds <= 86400):
            raise ValueError("worker duration must be positive and at most one day")
        if type(self.max_processes) is not int or not 2 <= self.max_processes <= 32:
            raise ValueError("worker process ceiling must be between 2 and 32, including interpreter launchers")
        if type(self.max_log_bytes) is not int or not 0 <= self.max_log_bytes <= 16 * 1024**2:
            raise ValueError("worker log ceiling must be between 0 and 16 MiB")


class WindowsJob:
    """An unnamed, non-inheritable job, including descendants without breakaway."""

    def __init__(self, limits):
        if os.name != "nt":
            raise NotImplementedError("OS-enforced worker memory limits currently require Windows; Linux containment is not implemented")
        import ctypes as C
        from ctypes import wintypes as W

        class BasicLimits(C.Structure):
            _fields_ = [("PerProcessUserTimeLimit", C.c_int64), ("PerJobUserTimeLimit", C.c_int64),
                        ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", C.c_size_t),
                        ("MaximumWorkingSetSize", C.c_size_t), ("ActiveProcessLimit", W.DWORD),
                        ("Affinity", C.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]

        class IOCounters(C.Structure):
            _fields_ = [(name, C.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount",
                "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimits(C.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
                        ("ProcessMemoryLimit", C.c_size_t), ("JobMemoryLimit", C.c_size_t),
                        ("PeakProcessMemoryUsed", C.c_size_t), ("PeakJobMemoryUsed", C.c_size_t)]

        class Accounting(C.Structure):
            _fields_ = [(name, C.c_int64) for name in ("TotalUserTime", "TotalKernelTime",
                "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")] + [
                (name, W.DWORD) for name in ("TotalPageFaultCount", "TotalProcesses",
                                           "ActiveProcesses", "TotalTerminatedProcesses")]

        self.C, self.ExtendedLimits, self.Accounting = C, ExtendedLimits, Accounting
        self.api = C.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([C.c_void_p, W.LPCWSTR], W.HANDLE),
            "SetInformationJobObject": ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
            "QueryInformationJobObject": ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p], W.BOOL),
            "IsProcessInJob": ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
            "TerminateJobObject": ([W.HANDLE, W.UINT], W.BOOL),
            "ResumeThread": ([W.HANDLE], W.DWORD),
            "CloseHandle": ([W.HANDLE], W.BOOL),
        }
        for name, (args, result) in signatures.items():
            method = getattr(self.api, name)
            method.argtypes, method.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise C.WinError(C.get_last_error())
        try:
            value = ExtendedLimits()
            # JOB_MEMORY, ACTIVE_PROCESS, KILL_ON_JOB_CLOSE. No BREAKAWAY flag.
            value.BasicLimitInformation.LimitFlags = 0x200 | 0x8 | 0x2000
            value.BasicLimitInformation.ActiveProcessLimit = limits.max_processes
            value.JobMemoryLimit = limits.max_committed_bytes
            self._check(self.api.SetInformationJobObject(self.handle, 9, C.byref(value), C.sizeof(value)))
        except BaseException:
            self.close()
            raise

    def _check(self, ok):
        if not ok:
            raise self.C.WinError(self.C.get_last_error())

    def verify_assignment(self, process):
        from ctypes import wintypes as W
        associated = W.BOOL()
        self._check(self.api.IsProcessInJob(process._handle, self.handle, self.C.byref(associated)))
        if not associated.value:
            raise RuntimeError("worker was not assigned atomically at process creation")

    def snapshot(self):
        value, accounting = self.ExtendedLimits(), self.Accounting()
        self._check(self.api.QueryInformationJobObject(self.handle, 9, self.C.byref(value), self.C.sizeof(value), None))
        self._check(self.api.QueryInformationJobObject(self.handle, 1, self.C.byref(accounting), self.C.sizeof(accounting), None))
        # On the tested Windows host, the peak also rose on an allocation that
        # returned MemoryError. Keep the raw OS counter, not an asserted maximum
        # of successfully allocated resident or committed memory.
        return {"os_peak_job_memory_bytes": value.PeakJobMemoryUsed,
                "active_processes": accounting.ActiveProcesses, "total_processes": accounting.TotalProcesses}

    def terminate(self):
        self._check(self.api.TerminateJobObject(self.handle, 1))

    def close(self):
        if self.handle:
            self._check(self.api.CloseHandle(self.handle))
            self.handle = None


class SuspendedPython:
    """Keep the primary thread handle that Popen normally closes immediately.

    The job list assigns containment atomically at process creation (Windows 10+
    required). There is no orphan window if the supervisor dies before resuming
    the primary thread. Only stdin=NUL and the log pipe handles are inherited.
    """
    def __init__(self, command, cwd, environment, job):
        import _winapi
        import msvcrt
        from ctypes import wintypes as W
        C, api = job.C, job.api

        class Startup(C.Structure):
            _fields_ = [("cb", W.DWORD), ("lpReserved", W.LPWSTR), ("lpDesktop", W.LPWSTR),
                ("lpTitle", W.LPWSTR)] + [(name, W.DWORD) for name in (
                "dwX", "dwY", "dwXSize", "dwYSize", "dwXCountChars", "dwYCountChars", "dwFillAttribute", "dwFlags")] + [
                ("wShowWindow", W.WORD), ("cbReserved2", W.WORD), ("lpReserved2", C.c_void_p),
                ("hStdInput", W.HANDLE), ("hStdOutput", W.HANDLE), ("hStdError", W.HANDLE)]

        class StartupEx(C.Structure):
            _fields_ = [("info", Startup), ("attributes", C.c_void_p)]

        class ProcessInfo(C.Structure):
            _fields_ = [("process", W.HANDLE), ("thread", W.HANDLE), ("pid", W.DWORD), ("tid", W.DWORD)]

        api.InitializeProcThreadAttributeList.argtypes = [C.c_void_p, W.DWORD, W.DWORD, C.POINTER(C.c_size_t)]
        api.InitializeProcThreadAttributeList.restype = W.BOOL
        api.UpdateProcThreadAttribute.argtypes = [C.c_void_p, W.DWORD, C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_void_p]
        api.UpdateProcThreadAttribute.restype = W.BOOL
        api.DeleteProcThreadAttributeList.argtypes = [C.c_void_p]
        api.DeleteProcThreadAttributeList.restype = None
        api.CreateProcessW.argtypes = [W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p, W.BOOL, W.DWORD,
                                     C.c_void_p, W.LPCWSTR, C.POINTER(StartupEx), C.POINTER(ProcessInfo)]
        api.CreateProcessW.restype = W.BOOL
        self.api = _winapi
        self.returncode = None
        self.command = command
        read_fd, write_fd = os.pipe()
        try:
            with open(os.devnull, "rb") as null:
                stdin = msvcrt.get_osfhandle(null.fileno())
                stdout = msvcrt.get_osfhandle(write_fd)
                os.set_handle_inheritable(stdin, True)
                os.set_handle_inheritable(stdout, True)
                startup, process_info = StartupEx(), ProcessInfo()
                startup.info.cb = C.sizeof(startup)
                startup.info.dwFlags = subprocess.STARTF_USESTDHANDLES
                startup.info.hStdInput, startup.info.hStdOutput, startup.info.hStdError = stdin, stdout, stdout
                size = C.c_size_t()
                api.InitializeProcThreadAttributeList(None, 2, 0, C.byref(size))
                if not size.value:
                    raise C.WinError(C.get_last_error())
                attributes = C.create_string_buffer(size.value)
                job._check(api.InitializeProcThreadAttributeList(attributes, 2, 0, C.byref(size)))
                try:
                    handles, jobs = (W.HANDLE * 2)(stdin, stdout), (W.HANDLE * 1)(job.handle)
                    job._check(api.UpdateProcThreadAttribute(attributes, 0, 0x20002, handles, C.sizeof(handles), None, None))
                    job._check(api.UpdateProcThreadAttribute(attributes, 0, 0x2000D, jobs, C.sizeof(jobs), None, None))
                    startup.attributes = C.cast(attributes, C.c_void_p)
                    block = C.create_unicode_buffer("\0".join(k + "=" + v for k, v in sorted(environment.items())) + "\0")
                    # NO_WINDOW | SUSPENDED | UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT
                    flags = subprocess.CREATE_NO_WINDOW | 0x4 | 0x400 | 0x80000
                    job._check(api.CreateProcessW(command[0], C.create_unicode_buffer(subprocess.list2cmdline(command)),
                        None, None, True, flags, block, str(cwd), C.byref(startup), C.byref(process_info)))
                    self._handle, self._thread, self.pid = process_info.process, process_info.thread, process_info.pid
                finally:
                    api.DeleteProcThreadAttributeList(attributes)
            self.stdout = os.fdopen(read_fd, "rb")
            read_fd = None
        finally:
            os.close(write_fd)
            if read_fd is not None:
                os.close(read_fd)

    def resume(self, job):
        value = job.api.ResumeThread(self._thread)
        if value == 0xFFFFFFFF:
            raise job.C.WinError(job.C.get_last_error())
        self.api.CloseHandle(self._thread)
        self._thread = None

    def wait(self, timeout):
        if self.api.WaitForSingleObject(self._handle, int(timeout * 1000)) != self.api.WAIT_OBJECT_0:
            raise subprocess.TimeoutExpired(self.command, timeout)
        self.returncode = self.api.GetExitCodeProcess(self._handle)
        return self.returncode

    def close(self):
        if self._thread:
            self.api.CloseHandle(self._thread)
            self._thread = None
        self.api.CloseHandle(self._handle)


def run_cpu_worker(python, arguments, *, cwd, log, limits, cancelled=lambda: False):
    """Contained trusted CPU worker; CUDA stays hidden regardless of caller env."""
    return _run_worker(python, arguments, cwd=cwd, log=log, limits=limits, cancelled=cancelled, gpu_research=False)


def run_gpu_research_worker(python, arguments, *, cwd, log, limits, allow_gpu=False, cancelled=lambda: False):
    """Explicit research only: RAM/time/tree containment, NOT a total VRAM quota.

    No DMN learning recipe calls this. The caller must obtain the maintenance
    agreement before opting in. Visibility alone does not constrain GPU memory.
    """
    if allow_gpu is not True:
        raise ValueError("GPU research requires explicit opt-in; CUDA remains unused")
    return _run_worker(python, arguments, cwd=cwd, log=log, limits=limits, cancelled=cancelled, gpu_research=True)


def _run_worker(python, arguments, *, cwd, log, limits, cancelled, gpu_research):
    """Run trusted Python code inside an OS-limited process tree; return evidence.

    The supervisor retains a job handle. It resumes the suspended child only
    after assignment succeeds. Closing that handle, including supervisor death,
    terminates descendants. A watchdog polls wall time/cancellation every 50 ms;
    this is bounded-response supervision, not a real-time deadline guarantee.
    No shell is used. Callers must select trusted code and validate its artifacts.
    """
    if not isinstance(limits, WorkerLimits):
        raise TypeError("validated WorkerLimits required")
    python, cwd, log = Path(python).resolve(), Path(cwd).resolve(), Path(log).absolute()
    if not python.is_file() or not cwd.is_dir():
        raise ValueError("worker interpreter and working directory must exist")
    if not isinstance(arguments, (list, tuple)) or not all(isinstance(arg, str) for arg in arguments):
        raise ValueError("worker arguments must be a sequence of strings")
    job = WindowsJob(limits)  # Fail before opening files or starting a worker.
    process, reader = None, None
    output = None
    started = time.monotonic()
    drained = {"bytes_seen": 0, "bytes_written": 0, "error": None}
    try:
        output = log.open("xb")
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1")
        environment.pop("DMN_GPU_PROBE_CONTAINED", None)
        if gpu_research:
            environment.update(CUDA_VISIBLE_DEVICES="0", DMN_GPU_PROBE_CONTAINED="1")
        process = SuspendedPython([str(python), *arguments], cwd, environment, job)

        def drain():
            # Keep draining after the cap: a noisy child must neither fill disk
            # nor deadlock on a full pipe. No accumulating communicate() buffer.
            try:
                while data := process.stdout.read(8192):
                    drained["bytes_seen"] += len(data)
                    kept = data[:max(0, limits.max_log_bytes - drained["bytes_written"])]
                    if kept:
                        output.write(kept)
                        drained["bytes_written"] += len(kept)
            except Exception as exc:
                drained["error"] = str(exc)

        reader = threading.Thread(target=drain, name="dmn-worker-log", daemon=True)
        reader.start()
        job.verify_assignment(process)
        if cancelled():
            outcome = "cancelled"
            job.terminate()
        else:
            process.resume(job)
            outcome = "exited"
            while True:
                # Track the entire job, including a child outliving its parent.
                snapshot = job.snapshot()
                if snapshot["active_processes"] == 0:
                    break
                if cancelled():
                    outcome = "cancelled"
                elif time.monotonic() - started >= limits.max_seconds:
                    outcome = "time_limit"
                elif drained["error"]:
                    outcome = "log_error"
                if outcome != "exited":
                    job.terminate()
                    break
                time.sleep(.05)
        process.wait(timeout=10)
        snapshot = job.snapshot()
        # Closing also kills any remaining descendants before joining the pipe.
        job.close()
        reader.join(timeout=10)
        if reader.is_alive():
            raise RuntimeError("worker log pipe did not close after job termination")
        output.flush()
        os.fsync(output.fileno())
        return {"outcome": outcome, "returncode": process.returncode,
                "succeeded": outcome == "exited" and process.returncode == 0 and not drained["error"],
                "elapsed_seconds": time.monotonic() - started, "limits": dataclasses.asdict(limits),
                "memory_enforcement": "windows_job_aggregate_commit", "wall_time_enforcement": "supervisor_watchdog",
                "disk_quota_enforced": False,
                "gpu_policy": ("explicit_gpu_research; no total process VRAM quota" if gpu_research else "trusted_cpu_code_and_hidden_cuda_devices"),
                **snapshot, "log_bytes_seen": drained["bytes_seen"], "log_bytes_written": drained["bytes_written"],
                "log_truncated": drained["bytes_seen"] > drained["bytes_written"], "log_error": drained["error"]}
    finally:
        # Cleanup must cover verification failure too: the assigned process is
        # suspended and has not executed the requested workload.
        job.close()
        if process:
            # Atomic job assignment makes CloseHandle sufficient. A second
            # TerminateProcess can race asynchronous job termination and fail
            # with ACCESS_DENIED even though the process is already exiting.
            process.wait(timeout=10)
            if reader:
                reader.join(timeout=10)
            if process.stdout:
                process.stdout.close()
            process.close()
        if output:
            output.close()
