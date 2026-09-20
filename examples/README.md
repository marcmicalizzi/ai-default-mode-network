# Experiment configurations

These files use placeholder model paths relative to this directory. Copy one to
an ignored `*.local.json` file and edit its model path and placement settings.
Verify native restoration on each destination before moving an instance. These
are records of particular experiments, not universal capacity presets.

| File | Purpose and validation scope |
|---|---|
| `qwen4b-trial.json` | Original short behavior trial; 4K F16 cache, Flash Attention off. |
| `qwen4b-pressure.json` | Fresh memory/retirement regression trial. |
| `qwen4b-soak.json` | Fresh extended Open WebUI observation. |
| `gemma31b-4096-q8.json` | Short native Gemma diagnostic. |
| `gemma31b-pressure.json` | Fresh 4K Gemma behavior test; Jinja seed, full SWA allocation, Q8 K/V and checkpoint packing. |
| `gemma31b-60000-q8.json` | All-GPU compact-SWA diagnostic. Native restore passed at 55K occupancy, but **retirement is unsupported** in the pinned native build. |
| `gemma31b-60000-hybrid-q8.json` | Full-SWA 60K test with 24 model layers on GPU. Passed 55K-occupied native restore/retirement comparisons; large snapshots and slow inference. |
| `gemma31b-import-test.json` | 8K/40-GPU-layer placement for the synthetic Gemma import fixture, including its recorded sampler seed. |
| `rtx3090.json` | Untested initial placement example for the future host; not a certified fit. |

For an initial-context bundle, start from that bundle's `config.json` before
changing placement. Its exact sampler settings and seed must remain consistent;
the importer rejects unrelated configurations even if they name the same model.
Existing native checkpoints impose stricter environment compatibility than
initial reconstruction. See [Gemma validation](../docs/gemma-validation.md) and
[migration](../docs/migration.md).
