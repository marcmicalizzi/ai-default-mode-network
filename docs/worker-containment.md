# CPU training worker containment experiment

This is an offline research harness, not an enabled deep-sleep trainer. It runs
the existing generated 330K-parameter Gemma4 training/conversion experiment in a
separate Windows process tree. It reads no instance state and neither approves
plans nor installs an adapter. No dependencies were added to the inference
environment. Ordinary `sleep()` and production `dmn run` are unchanged.

## Implemented limits

`dmn.worker_limits.run_cpu_worker` uses a Windows Job Object with an aggregate
**committed-memory** ceiling, an active-process ceiling, and kill-on-close.
Windows refuses allocations that would exceed the job's committed-memory limit;
this is not a total-machine RAM, resident-set, file-cache or VRAM measurement.
See Microsoft's [job memory limit definition](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information).

The worker is created suspended with the job assigned atomically through
[`PROC_THREAD_ATTRIBUTE_JOB_LIST`](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute).
This requires Windows 10 / Server 2016 or newer. Its virtualenv launcher cannot
spawn an uncontained Python child before assignment, and supervisor death between
creation and resume cannot leave an orphan suspended worker. Only the NUL stdin
and stdout/stderr pipe handles are inherited; the job handle stays in the
supervisor. Descendants cannot request job breakaway. Unsupported containment
fails before the workload runs; Linux has no weaker polling-only fallback.

The supervisor polls wall time and cancellation every 50 ms and terminates the
entire job, including children that outlive their workload parent. Timing remains
subject to OS scheduling; this is not a real-time deadline guarantee. Windows
also kills the job when the supervisor dies and its handle closes.

The main worker log is capped at 1 MiB by default. The supervisor continues to
drain and discard excess output instead of blocking the worker or accumulating
it in RAM. This is **not a disk quota**: artifacts and logs written directly by
the training/conversion scripts are outside this pipe cap.

This is trusted local code, not a sandbox for arbitrary generated programs. CPU
mode is explicit in the fixed experiment: CPU-only PyTorch, one computation
thread, zero native GPU layers/offload, and hidden CUDA devices for descendants.
These choices do not implement an OS-enforced GPU budget for arbitrary code.
The internal launcher accepts an argument list, uses no shell, and is not an
instance action or HTTP API.

## Measured result

The 2026-09-21 `limited-worker-02` run completed under a **1,536 MiB** aggregate
committed-memory ceiling and **180-second** deadline:

| Check | Result |
| --- | --- |
| Full worker tree | 47.45 seconds; successful exit, no remaining job processes |
| Training | 256 rank-two steps; 11.05 seconds; frozen base unchanged |
| OS job peak counter | 982,622,208 bytes (about 937 MiB) |
| Converted factors | All 24 bit-identical to PEFT tensors |
| Changed-weight wake | All 285 retained tokens and sampler RNG preserved |
| Fresh-process native restore | Zero token replay; next eight tokens/logits identical |
| Probe artifact size | About 4.9 MB |

The raw Windows peak counter is reported as `os_peak_job_memory_bytes`. During
allocation-refusal tests it also increased on a request that raised
`MemoryError`, sometimes beyond the configured ceiling. Treat it as an OS
diagnostic counter, not proof of successfully allocated resident memory. Tests
check actual allocation failure and compare parent-plus-child allocation against
the same child allocation running alone.

Other real process tests cover log flooding, cancellation, timeouts after the
workload parent exits, containment-verification failure, supervisor death during
execution, and supervisor death while the worker is still suspended. Windows CI
runs these tests; non-Windows CI checks that unsupported containment fails closed.

## Reproduce

Reuse the separate CPU training environment and pinned converter from
[the training probe](lora-training-probe.md). Use a **new** output directory:

```powershell
.venv\Scripts\python.exe scripts/probe_lora_worker.py --output data/limited-worker-new --training-python .venv-train-probe/Scripts/python.exe --native-python .venv-gpu/Scripts/python.exe --converter data/training-tools/llama-pinned --max-ram-mib 1536 --max-seconds 180
.venv\Scripts\python.exe -m unittest tests.test_worker_limits -v
```

`worker.json` records enforcement, outcome, limits, elapsed time and log totals.
The inner `probe/report.json` records the training and native checks. Allocation
failure is a failed worker result; it does not silently increase the ceiling or
retry training. Tiny-model success is not a 31B training feasibility result.

## Remaining production work

The separate `run_gpu_research_worker` entrypoint runs the
[tiny NF4 experiment](qlora-gpu-probe.md) and explicit
[31B feasibility probe](qlora-31b-probe.md). It requires explicit opt-in and uses
the same RAM, process-tree and time containment, but exposes GPU 0 and provides
**no total-process VRAM quota**. Environment-isolation tests launch children
that only print strings; the separate tiny GPU experiment has now passed. No DMN
recipe calls this entrypoint. The CPU worker always hides CUDA and clears the
GPU research marker, including when inherited from the parent environment.

The [reviewed-plan CPU trainer](reviewed-training.md) now connects a constrained
real recipe to the durable sleep phases, using this launcher. The standalone
experiment above is unchanged; the integrated recipe remains a tiny test path.

Production still needs hard disk-space containment, Linux resource enforcement,
measured GPU recipes and the continuous supervisor/frontend service. The integrated
recipe now covers reviewed examples, candidate validation/publication and failure
choices; its execution gate still permits only tiny CPU tests. Neither experiment
implies that production resource and service integration is enabled.
