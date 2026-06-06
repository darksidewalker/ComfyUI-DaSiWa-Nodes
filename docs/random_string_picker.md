# DaSiWa Random String Picker

**Category:** `DaSiWa/Text`  
**Class name:** `DaSiWa_RandomStringPicker`  
**File:** `nodes/random_string_picker.py`

---

## Overview

The Random String Picker is a text bridge node for prompt variation. It accepts a connected `STRING` input, replaces each complete `{A|B|C}` group with one randomly selected option, and outputs the processed text as a `STRING`.

Everything outside complete `{...}` groups passes through unchanged.

---

## Inputs

| Input | Type | Description |
|---|---|---|
| `text` | STRING | Connected text input. Use `{A|B|C}` syntax anywhere in the text to define random choices. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `text` | STRING | The input text with each complete random group replaced by one selected option. |

---

## Syntax

Use curly braces for a random group and separate options with `|`.

```text
A {red|blue|green} car in {sunlight|rain|fog}
```

Possible output:

```text
A blue car in fog
```

Each group is processed independently, so a prompt can contain any number of random groups:

```text
{wide angle|macro} photo of a {cat|robot|flower}, {soft light|hard shadows}
```

Possible output:

```text
macro photo of a robot, soft light
```

---

## Pass-through behavior

Only complete groups matching `{...}` are changed. Text before, after, and between groups remains unchanged.

```text
prefix {A|B} middle {X|Y} suffix
```

Possible output:

```text
prefix B middle X suffix
```

The words `prefix`, `middle`, and `suffix`, including spaces and punctuation outside the groups, are preserved.

Unmatched braces are treated as normal text because they do not form a complete random group.

```text
This {will not change
```

Output:

```text
This {will not change
```

---

## Notes

- Empty options are allowed. For example, `{small||large}` can output `small`, an empty string, or `large`.
- Nested braces are not parsed as nested random groups.
- The node marks itself changed on each run so ComfyUI will re-evaluate the random choices when queued again.
