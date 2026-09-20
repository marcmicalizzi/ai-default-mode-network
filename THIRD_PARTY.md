# Third-party software and model licenses

The [MIT license](LICENSE) covers this project's original code. This repository
does not vendor llama.cpp, llama-cpp-python, Open WebUI, or model weights. Installing
them separately does not relicense them under this project's license.

| Component | Upstream terms | Use here |
|---|---|---|
| llama.cpp | [MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) | Native inference engine accessed through the Python binding |
| llama-cpp-python 0.3.35 | [MIT](https://github.com/abetlen/llama-cpp-python/blob/v0.3.35/LICENSE.md) | Separately installed optional dependency |
| Open WebUI 0.11.0 | [Open WebUI License](https://github.com/open-webui/open-webui/blob/v0.11.0/LICENSE), with [historical terms](https://github.com/open-webui/open-webui/blob/v0.11.0/LICENSE_HISTORY) | Optional, separately installed frontend; original Pipe/Event adapters and version-checked runtime hooks |

Copies or substantial portions of MIT-licensed dependencies must retain their
copyright and permission notices. Open WebUI has its own notice, endorsement and
branding conditions, including stated exceptions; consult the linked license for
the exact terms. The adapter leaves Open WebUI branding in place. Its internal
hooks are not a fork or a copy of the upstream implementation.

If distributing a combined installer, container, binary, or modified upstream
version, review the licenses and notices of everything included, including
transitive dependencies. This page is an inventory, not a substitute for those
licenses. Changing dependency versions can change the applicable obligations.

Model weights have separate licenses and sometimes acceptable-use terms.
Naming a GGUF in an example does not grant permission to use or redistribute it;
check both the underlying model's and the distributor's terms. Downloaded models,
private conversations, and runtime state are excluded from this repository.
