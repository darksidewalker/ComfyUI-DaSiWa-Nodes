# Registry Security Review

This document records the Comfy Registry findings and the data-flow review for the public node package. It does not contain raw scanner payloads, source snippets, credentials, or unpublished release details. The former test suites were removed from this repository; any test evidence named below is historical, not a current validation gate.

## Registry evidence

On 2026-09-15, `python tools/audit_comfy_registry_status.py` showed that `0.4.22` is active and every release from `0.4.23` through `0.4.37` is banned. Versions `0.4.31` through `0.4.34` identify an actual remote-code path in the LLM selector: workflow-controlled Hugging Face repository download plus `trust_remote_code`. The remediation removes those controls and locks Ollama to loopback.

The historical `policy-v0.4: rce-remote-code` scan also reported static findings. They require an explicit decision or a manual review; they must not be obscured to alter scanner results.

The latest public version checked, `0.5.5`, is `NodeVersionStatusFlagged` with nine informational YARA findings. Its reason is a direct JSON list, not the policy/history object returned for `0.5.3`; the audit tool now supports both formats. Version `0.5.4` is pending. Version `0.5.3` also carries `policy-v0.5: arbitrary-file-read`, without a specific vulnerable source location in the policy message.

## Local remediation awaiting publication and review

- Replace constant-name dynamic imports with ordinary imports. RTX cleanup uses the native `comfy.model_management` module.
- Split the Director log statement across adjacent strings without changing its output.
- Resolve preview source symlinks before output-root containment checks. A linked file or directory cannot select a source outside the resolved output root.
- Constrain LoRA metadata and sidecar image routes to resolved, operator-configured LoRA roots. Reject absolute paths, traversal and symlinks escaping those roots. Additional configured roots and symlinked root directories remain supported. A model-only symlink to an unregistered external directory no longer grants HTTP access; configure that directory as a LoRA root instead.

Local regression probes reproduced the audit parsing failure and symlink escapes before the fixes. They cover rejected paths and retained nested/root-alias behavior. These findings are not proof that the Registry's older `arbitrary-file-read` decision refers to the same paths. The changes are not an accepted Registry release.

The JavaScript `Function.bind` network finding, environment-variable reads and fixed Civitai requests still require manual review. Retain those capabilities rather than disguising them to suppress scanner matches.

## Finding inventory

| Finding | Location | Data origin | Current guard | Impact if guard fails | Decision | Verification |
| --- | --- | --- | --- | --- | --- | --- |
| Remote model execution | `nodes/nodes_llm.py` (removed download and remote-code controls) | `/prompt` widget values | Previously none sufficient | Attacker-selected repository code could execute in the ComfyUI process | Removed | Historical test evidence; Registry audit after release |
| Arbitrary server-side request via Ollama URL | `nodes/nodes_llm.py` (removed URL widget) | `/prompt` widget value | Previously none | Server-side request to attacker-chosen service | Removed; endpoint is fixed to loopback | Historical test evidence; review request destination manually |
| PyAV media processing | `nodes/nodes_enhanced_video_combine.py`, `nodes/helper_pyav_video.py` | Request supplies a basename and subfolder for an output asset; node execution supplies image/audio tensors and an output path | PyAV 18 calls its bundled FFmpeg libraries in-process; no executable lookup, shell, or subprocess remains. Only `type=output` is accepted by preview; basename/traversal/resolved-commonpath checks constrain its source to the ComfyUI output root | A vulnerability in PyAV or its native FFmpeg libraries could affect the ComfyUI process | Retain with bounded inputs and manual review | Historical test evidence; manually review preview-root guard and encode fallback |
| Civitai by-hash request | `nodes/lora_info.py` (`fetch_civitai`, `lora_info`) | A selected local LoRA name resolves through `folder_paths`; SHA-256 becomes the Civitai request key | LoRA path resolution uses `folder_paths` plus resolved-root containment; request destination is a module constant | Local file metadata is sent as a deterministic hash to a third party; a route can trigger outbound traffic | Retain with disclosure and manual review | Historical test evidence; manually review by-hash destination and cache |
| Hardware telemetry subprocess | `nodes/nodes_system_monitor.py:67-73,76-138,179-205` | Fixed internal GPU probe lists | Executable is looked up by fixed name; arguments are constants; no shell | A future command interpolation regression could run unintended programs | Retain for manual review | Historical test evidence; manually review fixed-command arguments |
| Constant-name dynamic imports | Package registration, Forge, LLM transport and RTX cleanup | Constant module names | Ordinary imports; RTX uses `comfy.model_management` | The old patterns triggered import-evasion rules without workflow-controlled module selection | Replaced locally | Focused Forge/LLM tests; CPU package registration and native RTX module identity check |
| Long Director log statement | `nodes/nodes_minimax_h3_director.py` | Internal model/timeline values | No code loading or process launch | None; scanner misclassifies semicolon density inside a string as minification | Split into adjacent strings locally | Original and edited module ASTs are identical |

## Manual-review request

The package retains the bounded FFmpeg, Civitai, and telemetry capabilities. Manual-review evidence was submitted to [Comfy Registry backend issue #234](https://github.com/Comfy-Org/registry-backend/issues/234). If reviewers identify a specific policy violation, replace that capability rather than attempting cosmetic scanner evasion. The final release is accepted only when the Registry API reports the exact version as `NodeVersionStatusActive`.
