# Wildcard & Preset Prompt Builder

`DaSiWa Wildcard & Preset Prompt Builder` builds independent positive and negative `STRING` prompts from an editable library. It is designed to work without a downstream wildcard picker.

## Node inputs and controls

- **positive_input** / **negative_input**: optional connected `STRING` inputs. When present, each value is prepended to that side's generated result. Blank values are ignored. The text may contain `$name` prompt variables (see Prompt Variables below).
- **variables_json**: JSON object mapping variable names to lists of strings, resolved inside `positive_input`/`negative_input` — see Prompt Variables below.
- **variable_selections**: JSON object pinning single occurrences, e.g. `{"hairstyle:0": "bob cut"}` — see Prompt Variables below.
- **Seed**: makes wildcard alternatives reproducible.
- **🎲 New Picks**: changes the stored reroll value while retaining the selection, weights, style, and seed.
- **🎲 Random Select**: replaces the current selection with a secure-random set of 1–10 available subjects, drawn from both Presets and Wildcards for the current style. It preserves a prior weight only when that same subject is selected again; new selections use `1.0`.
- **New picks on every queue**: opts into a fresh random wildcard choice for every queued execution. It does not change the saved manual reroll state.
- **Token budget**: estimates token count locally and removes complete enabled subjects, lowest weight first, until the generated side fits the limit. Positive and negative prompts have independent limits.
- **Style**: chooses the `booru_*` or `nl_*` library fields.

Select a subject independently for the positive or negative side. A weight of `1.0` emits plain text; another value emits ComfyUI emphasis syntax such as `(cinematic lighting:1.2)`.

## Selection visibility

The category list and prompt previews remain in a compact scroll panel in both the classic renderer and Nodes 2.0. Click inside the panel to focus it before using the mouse wheel; modified wheel gestures remain available for ComfyUI's canvas navigation.

Each Presets/Wildcards category header shows a green, right-aligned `✓ N selected` badge when it contains enabled subjects, and the category border is highlighted. This stays visible when the category is collapsed, making Random Select results and manual selections easy to find. The badge count applies to that section only: a preset and wildcard with the same subject name are independent selections.

## Prompt Variables

`positive_input` and `negative_input` text can contain `$name` tokens that resolve from your own JSON library **before** the generated presets are prepended.

```json
{"hairstyle": ["bob cut", "long wavy", "ponytail"], "color": ["crimson", "azure"]}
```

- `$name` (or `${name}` to delimit, e.g. `${color}ball`) is replaced by a library entry. `$$` renders a literal `$`. Unknown names stay in the text untouched.
- Every occurrence resolves independently: the same variable twice in one prompt yields two independent picks (occurrence keys are `name:index`).
- Picks follow the node **Seed**: same seed + same library = same prompt. *New picks on every queue* re-rolls the generated presets only — variable picks stay stable with the seed.
- Pin an occurrence by adding `variable_selections`: `{"hairstyle:0": "ponytail"}`. Pinned values beat the seed pick and survive seed changes.
- Names must be letters/digits/underscores (not digit-leading), and every list needs at least one non-blank string.

## HTTP routes

- `GET /dasiwa/wildcard-preset-prompt-builder/library` returns the library JSON.
- `POST /dasiwa/wildcard-preset-prompt-builder/build` takes the node's inputs as JSON (`style`, `seed`, `reroll`, `token_budget`, `selection_state`, optionally `positive_input`, `negative_input`, `variables_json`, `variable_selections`) and returns `{"positive", "negative"}` resolved exactly as the node would with that reroll — inputs with `$vars` resolved and presets prepended — plus `positive_base`/`negative_base` (the resolved input text without presets). DEAS uses it to seed the Prompt Writer with the active wildcards and to preview variable resolution.

## Library file

The bundled starter library is:

`data/wildcards_and_presets_dual.json`

It is ordinary JSON. You may edit it, replace it, or paste in a compatible custom version. There is **no required version field, checksum, sidecar, or hash-generation step**. Keep a backup of your custom file when updating the node, because a package update may replace the bundled starter file.

Restart ComfyUI after changing the library so the node reloads it.

## JSON structure

The root object must contain `wildcard_library.categories`. Categories are an object, and their JSON order is the prompt assembly order. Each category contains an array of subject objects.

```json
{
  "wildcard_library": {
    "categories": {
      "Positive Prompts": [
        {
          "subject": "Detail Level",
          "booru_wildcards": ["{high detail|very detailed}"],
          "nl_wildcards": ["{highly detailed|rich in fine detail}"],
          "booru_presets": [],
          "nl_presets": []
        }
      }
    }
  }
}
```

### Subject fields

| Field | Meaning |
| --- | --- |
| `subject` | Required visible name in the node UI. |
| `booru_wildcards` | Optional list of Booru/tag-style prompt fragments. One alternative is resolved from every `{option A|option B}` group. |
| `nl_wildcards` | Optional list of natural-language prompt fragments. |
| `booru_presets` | Optional fixed Booru/tag-style prompt fragments. |
| `nl_presets` | Optional fixed natural-language prompt fragments. |

Use `[]` for a field with no content. A subject may provide wildcards, presets, or both. The UI only shows a subject in the relevant Wildcards or Presets section when that section has content for the currently selected style.

The shipped file also includes legacy `wildcards` and `presets` arrays on its entries for compatibility with older tools. The builder selects from the style-specific `booru_*` and `nl_*` fields, so custom library entries should provide those fields.

## Character Selection presets

`Character Selection` is a preset-only category placed after Art Style and before species details. It provides fixed, individually selectable Booru count tags for `1girl` through `5girls`, `1boy` through `5boys`, `1man` through `5men`, and `1woman` through `5women`, plus `solo`, `group`, and `crowd`. Natural Language emits proper phrases instead: `1girl` becomes `a girl`, `2boys` becomes `two boys`, and `5men` becomes `five men`.

Each count is its own preset subject—not a wildcard—so selecting `3 Girls` always emits `3girls`; it never randomly changes to another count. Select one count/identity subject and, when useful, one scene-size subject such as `Solo`, `Group`, or `Crowd`. The builder does not automatically prevent contradictory manual selections.

## Prompt text rules

- `{a|b|c}` picks exactly one choice every time that subject is resolved.
- Plain text outside braces is retained verbatim: `soft {blue|red} lighting` becomes `soft blue lighting` or `soft red lighting`.
- Entries within one selected subject are joined with commas.
- Enabled subjects are joined in category order.
- Positive and negative selections are separate; selecting a subject on one side does not enable it on the other.
- Optional connected `positive_input` and `negative_input` values are placed before the generated text on their respective sides, after their `$name` prompt variables are resolved.

## Bundled category order

The bundled file uses an order intended to produce a model-friendly prompt: broad positive quality and style first, then subject/species, character attributes, clothing/body framing, pose/composition, scene, optional adult content, and negative controls last.

1. Positive Prompts
2. Art Style
3. Character Selection
4. Species - General
5. Species - Mythical
6. Species - Botanical
7. Species - Kemonomimi
8. Species - Mammals
9. Species - Aquatic
10. Species - Reptiles, Amphibians & Birds
11. Species - Insects & Arachnids
12. Character - Eyes
13. Character - Face
14. Character - Hair
15. Character - Body
16. Wardrobe
17. Body Visibility
18. Composition & Pose
19. Scene
20. NSFW
21. Negative Prompts

`Species - Botanical` contains only botanical concepts. Fantasy, undead, and creature concepts belong in `Species - Mythical` or another relevant species category.
