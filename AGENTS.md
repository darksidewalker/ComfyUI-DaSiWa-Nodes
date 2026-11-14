# Nodepack development rules

## DEAS compatibility gate

Check every nodepack change against DEAS before calling it complete or publishing it. This nodepack is also consumed by DEAS; preserve standalone ComfyUI behavior as well.

- Inspect the affected node IDs, inputs, outputs, routes and helpers in the nodepack, then search the DEAS checkout for their consumers or copied logic. Use the sibling `DEAS` repository when available.
- For affected DEAS integrations, trace the actual emitted graph or request and run the focused nodepack regression tests plus the relevant DEAS compiler/consumer tests. For Core/model-management changes, also check the actual DEAS-owned ComfyUI source and exercise the regression against that Core when available.
- Preserve saved-workflow and API compatibility. Do not add DEAS-specific behavior to standalone execution or duplicate a shared nodepack fix in DEAS unnecessarily.
- If DEAS has no affected consumer, record that finding. If its checkout, runtime or hardware is unavailable, name the missing verification explicitly; do not claim cross-project acceptance from nodepack-only tests.
- Report source/test verification separately from real model inference and installed-runtime verification. A published nodepack fix is not an updated DEAS installation; users receive it through Backend → Upgrade → ComfyUI and node packs, followed by a backend restart.
- Do not silently patch installed nodepacks, restart services or include unrelated DEAS working-tree changes. Cross-repository edits and publication must remain within the user's authorized scope.
