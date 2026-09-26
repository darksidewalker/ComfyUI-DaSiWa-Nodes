# Registry Security Review

This document records the Comfy Registry findings and the data-flow review for the public node package. It does not contain raw scanner payloads, source snippets, credentials, or unpublished release details. The former test suites were removed from this repository; any test evidence named below is historical, not a current validation gate.

## Registry evidence

On 2026-09-15, `python tools/audit_comfy_registry_status.py` showed that `0.4.22` is active and every release from `0.4.23` through `0.4.37` is banned. Versions `0.4.31` through `0.4.34` identify an actual remote-code path in the LLM selector: workflow-controlled Hugging Face repository download plus `trust_remote_code`. The remediation removes those controls and locks Ollama to loopback.

The Registry's current `policy-v0.4: rce-remote-code` scan also reports static findings. They require an explicit decision or a manual review; they must not be obscured to alter scanner results.

## Finding inventory

| Finding | Location | Data origin | Current guard | Impact if guard fails | Decision | Verification |
| --- | --- | --- | --- | --- | --- | --- |
| Remote model execution | `nodes/nodes_llm.py` (removed download and remote-code controls) | `/prompt` widget values | Previously none sufficient | Attacker-selected repository code could execute in the ComfyUI process | Removed | Historical test evidence; Registry audit after release |
| Arbitrary server-side request via Ollama URL | `nodes/nodes_llm.py` (removed URL widget) | `/prompt` widget value | Previously none | Server-side request to attacker-chosen service | Removed; endpoint is fixed to loopback | Historical test evidence; review request destination manually |
| PyAV media processing | `nodes/nodes_enhanced_video_combine.py`, `nodes/helper_pyav_video.py` | Request supplies a basename and subfolder for an output asset; node execution supplies image/audio tensors and an output path | PyAV 18 calls its bundled FFmpeg libraries in-process; no executable lookup, shell, or subprocess remains. Only `type=output` is accepted by preview; basename/traversal/commonpath checks constrain its source to the ComfyUI output root | A vulnerability in PyAV or its native FFmpeg libraries could affect the ComfyUI process | Retain with bounded inputs and manual review | Historical test evidence; manually review preview-root guard and encode fallback |
| Civitai by-hash request | `nodes/lora_info.py:179-191,203-266` | A selected local LoRA name resolves through `folder_paths`; SHA-256 becomes the Civitai request key | LoRA path resolution uses `folder_paths`; request destination is a module constant | Local file metadata is sent as a deterministic hash to a third party; a route can trigger outbound traffic | Retain with disclosure and manual review | Historical test evidence; manually review by-hash destination and cache |
| Hardware telemetry subprocess | `nodes/nodes_system_monitor.py:67-73,76-138,179-205` | Fixed internal GPU probe lists | Executable is looked up by fixed name; arguments are constants; no shell | A future command interpolation regression could run unintended programs | Retain for manual review | Historical test evidence; manually review fixed-command arguments |
| Dynamic import | `nodes/nodes_rtx_upscaler_refiner.py:23-29` | Constant module name | The module name is not workflow input | Scanner treats dynamic import as import evasion; compatibility behavior is not yet covered | Replace with a normal guarded import after manual compatibility review | Manual review pending |
| Long Director log statement | `nodes/nodes_minimax_h3_director.py:236` | Internal model/timeline values | No code loading or process launch | None; scanner misclassifies semicolon density as minification | Reformat only after manual behavior review or include in manual-review evidence | Manual review pending |

## Manual-review request

The package retains the bounded FFmpeg, Civitai, and telemetry capabilities. Manual-review evidence was submitted to [Comfy Registry backend issue #234](https://github.com/Comfy-Org/registry-backend/issues/234). If reviewers identify a specific policy violation, replace that capability rather than attempting cosmetic scanner evasion. The final release is accepted only when the Registry API reports the exact version as `NodeVersionStatusActive`.
