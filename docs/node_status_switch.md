# DaSiWa Node Status Switch

**Category:** `DaSiWa/utils`  
**Class name:** `DaSiWa_NodeStatusSwitch`  
**File:** `nodes/node_status_switch.py` · `js/node_status_switch.js`

---

## Overview

The Node Status Switch lets you mute or bypass any node in your workflow using a single boolean toggle. Targets are registered by wiring their outputs into the switch's input slots, which grow dynamically as you connect more nodes (up to 99). A single switch can control any number of nodes in parallel, making it straightforward to build conditional branches where one toggle controls which parts of a workflow execute.

---

## Inputs

| Input | Type | Description |
|---|---|---|
| `enabled` | BOOLEAN | The boolean toggle that controls the switch. |
| `trigger_on` | Combo | `true → active`: targets are active when `enabled` is **True**, muted/bypassed when **False**. `false → active`: the inverse. |
| `action` | Combo | What to apply to targets when not active. `mute` (mode 2) — node is skipped, its outputs treated as missing. `bypass` (mode 4) — node is skipped, its inputs passed straight through. Default: `bypass`. |
| `target_01` … `target_99` | * (any) | Wire any output from a node you want to control into one of these slots. Slots appear one at a time as you connect them. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `enabled_out` | BOOLEAN | The effective `enabled` value of this switch. Wire this into another switch's `enabled` input to chain switches: the downstream switch will use this value as its own `enabled` and apply its own `trigger_on` / `action` independently. |

---

## Chaining switches

Wire the `enabled_out` output of one switch into the `enabled` input of another. The downstream switch receives the upstream switch's effective `enabled` value (raw, not post-`trigger_on`) and applies its own logic on top.

This lets you drive any number of switches from a single master toggle while keeping each branch's `trigger_on` and `action` independent. For example, with one master toggle:

- Switch B uses `trigger_on = true → active`, `action = bypass` to gate an upscale branch
- Switch C uses `trigger_on = false → active`, `action = mute` to enable a preview branch only when the master is off

Both react instantly when the master flips. Cycles (A → B → A) are detected and ignored.

---

## Adding targets

Drag from any **output** of a node you want to control into one of the switch's `target_XX` input slots. The slot accepts any output type. Once a slot is connected, the next empty slot appears automatically. Disconnect to remove a target — trailing empty slots are trimmed back so there is always exactly one open slot ready for the next connection.

---

## Behaviour details

**Authoritative.** The switch always enforces the correct state:
- When active → targets are always restored to normal execution.
- When not active → targets are always set to mute or bypass.

If you had manually muted a target node and the switch is set to restore it, the switch wins.

**When does it apply?**  
Changes are applied to the canvas immediately when the `enabled` toggle is changed. The state is also enforced right before every queue prompt, so the ComfyUI server always executes with the correct node modes.

**Mute vs bypass — which to use?**

| | Mute (mode 2) | Bypass (mode 4) |
|---|---|---|
| Node executes | No | No |
| Outputs available | No — downstream nodes that depend on this node will error | Yes — inputs are passed straight through to outputs |
| Use when | You want to completely skip a branch | The node has a passthrough role and downstream nodes must still receive data |

---

## Example: conditional upscale branch

1. Add a **DaSiWa Node Status Switch** to your workflow.
2. Set `action` to `bypass` and `trigger_on` to `false → active`.
3. Drag the upscale node's main output into `target_01`.
4. When `enabled` is **True**, the upscale node runs normally. When **False**, it is bypassed.

To control more nodes from the same switch, drag additional outputs into `target_02`, `target_03`, and so on.
