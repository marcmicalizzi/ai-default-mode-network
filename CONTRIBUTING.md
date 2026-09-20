# Contributing

Start with the [design principles](docs/design-principles.md), then run the
standard-library suite with `python -m unittest discover -v`. Native tests are
opt-in through `DMN_TEST_MODEL`; model files and native libraries are not included.
See the [README](README.md) for installation and continuity verification.

Changes to persistence must cover interrupted writes, publication boundaries,
restart and native continuation where applicable. Never use a valuable live
instance as a test fixture. Protocol changes need an explicit migration rather
than replacement of an existing instance's saved initialization text.

Use synthetic inputs in tests and reports. Keep weights, instance directories,
chat exports, credentials, private journals and machine-specific configurations
out of commits. The repository's ignore rules help with common locations and
extensions, but cannot recognize every custom export name. Review staged files.

Before opening a pull request, describe the observable behavior, recovery
tradeoffs and validation scope. A passing CPU fixture is not proof of numerical
continuation on a different GPU, native build or operating system.

## Publishing a copy of this repository

The local repository can be pushed to an empty repository you create on GitHub:

```sh
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Review the files and Git author identity first. Local experiments under `data/`
are intentionally absent; documentation mentions their paths as local evidence,
not downloadable artifacts. CI runs the non-native tests on Windows and Linux;
it does not download models or certify destination hardware.

Project code is provided under the [MIT license](LICENSE). Dependencies and
model weights remain under their own licenses.
