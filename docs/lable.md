# Lable (DaSiWa)

A workflow-only annotation included in DaSiWa Custom Nodes from version **0.4.67**. It replaces the role of an rgthree Label without requiring rgthree. It has no execution inputs or outputs, no backend node or server route, and no extra runtime dependencies.

## Add and edit

Add **Lable (DaSiWa)** under **DaSiWa / utilities**. Double-click its body or choose **Edit label…** from the context menu. Changes apply immediately; **Done** or Escape closes the editor without undoing edits.

Drag the unpinned label by its body. The pointer stays a crosshair over text, images, and background. Use native resize handles or the Width and Height controls. **Fit to text** sizes the label for unwrapped text, without reserving space for images. Pin prevents dragging and forwards left pointer events to the graph underneath; right-click a visible pinned label to edit or unpin it.

## Text and appearance

- Multiline text; literal `\n` also starts a new line.
- Eight font choices, each previewed in its own font where the browser supports styled select menus. The selected font is previewed in the control. Missing fonts use browser fallback.
- Font size, left/center/right alignment, independent text and background opacity, padding, corner radius, and rotation.
- Every slider has an editable number. Press Enter or leave the field to apply it; values are constrained to the slider's range.
- The settings dialog is 680 pixels wide, constrained to the viewport. Small screens retain scrolling rather than hiding controls.

## Colors

Text and background each offer 48 palette swatches, a native RGB color picker, an editable HEX field, and an eyedropper icon.

HEX accepts `#RRGGBB` and shorthand `#RGB`, with or without the leading `#`. Enter or leaving the field applies the value. Invalid input restores the previous color and displays an English message. Palette, picker, HEX, and eyedropper values stay synchronized.

ComfyUI's native node-color menu changes the label background color but preserves its opacity. Opacity **0** remains transparent. **No color** clears the background. Native canvas decoration is kept transparent, including when restoring older workflows that saved opaque node colors.

Screen sampling uses the browser's **EyeDropper API**, normally available in desktop Chromium on localhost or HTTPS. Unsupported browsers disable the icon; HEX, palette, and native picker remain usable. Physical screen sampling may require desktop permission and is not verified by headless tests.

## Embedded images

**Choose image…** selects a local PNG, JPEG, or WebP, up to **10 MiB** per image. The image is embedded in workflow JSON, not uploaded or referenced by an external URL. It travels with the workflow but increases its size.

Image position choices:

- **background:** image scales to the full label bounds behind the text.
- **float left / float right:** text wraps around the image, then continues across the label width.
- **above text / below text:** image sits before or after the text.

**Contain** preserves the entire image; **cover** crops to fill its image box. Image width and opacity have sliders and editable numbers. **Remove image** clears the embedded image. Text and rotated content are clipped to label dimensions; enlarge the label if necessary.

Node-owned controls and tooltips are English. Native browser color dialogs and OS file dialogs follow browser/system language. User-provided text and filenames are not translated.

## Workflow compatibility

Labels save their title, properties, and embedded images in workflow JSON. Existing rgthree labels are not automatically converted.

The extension uses its own `DaSiWa.Lable` registration name and label-scoped styles, and skips registering its node type when it already exists. It does not modify rgthree, other node classes, or global canvas rendering.

## Compatibility and verification

Browser-tested with ComfyUI **0.38.0**, frontend **1.53.6**, in classic canvas and Nodes 2.0. The implementation uses `registerCustomNodes`, a virtual `LGraphNode`, and `addDOMWidget`, served from the pack's existing `WEB_DIRECTORY = "./js"`. A few scoped host selectors may require adjustment after frontend layout changes.

The label is deliberately absent from `/object_info` and API prompts: it annotates the browser graph, not backend execution.

With an isolated ComfyUI serving this pack, install the test-only packages Playwright and Pillow and run:

```sh
python .github/scripts/label_browser_smoke.py http://127.0.0.1:8199
```

`LABEL_TEST_CHROME` can select an existing Chromium executable. `LABEL_TEST_ASSET` can override the asset URL when the pack is loaded under a temporary directory name. Tests exercise both renderers, duplicate-registration protection, dragging, cursor style, font previews, numeric controls, RGB/HEX synchronization, invalid HEX, native-color opacity, all three image formats and five layouts, workflow restoration, pin/click-through, prompt exclusion, unsafe image rejection, and cleanup. EyeDropper results are explicitly stubbed; the test checks integration, not real monitor sampling.
