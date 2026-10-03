// Upgrade saved workflows without changing current v3 source/Duration semantics.
import assert from "node:assert/strict";
import { test } from "node:test";

const { normalizeContinuity, continuityTiming } = await import(new URL("../../js/minimax_h3_continuity.js", import.meta.url));

test("active v2 keeps exact frame duration and migrates once", () => {
  const duration = { value: 5 };
  const c = normalizeContinuity({ version: 2, operation: "continue", source_id: "parent",
    extension_frames: 238, continuation_prompt: "Turn left.", idea: "Camera follows." }, duration);
  assert.equal(c.version, 3);
  assert.equal(c.source_id, "parent");
  assert.equal(c.operation, "continue");
  assert.equal(duration.value, 238 / 24);
  assert.equal(continuityTiming(duration.value).extension_frames, 238);
  assert.equal(c.continuation_prompt, "Turn left.\nCamera follows.");
  assert.equal(c.idea, "");
  assert.equal("extension_frames" in c, false);
  duration.value = 15;
  normalizeContinuity(c, duration);
  assert.equal(duration.value, 15);
  assert.equal(c.continuation_prompt, "Turn left.\nCamera follows.");
});

test("inactive legacy source stays inactive and capture preference survives", () => {
  const duration = { value: 10 };
  const c = normalizeContinuity({ version: 1, operation: "new", source_id: "old-parent",
    source_video_id: "old-upload", source_video: { filename: "old.mp4" }, capture: false,
    extension_frames: 119, continuation_prompt: "A new scene." }, duration);
  assert.equal(c.operation, "new");
  assert.equal(c.source_id, "");
  assert.equal(c.source_video_id, "");
  assert.equal(c.source_video, null);
  assert.equal(c.capture, false);
  assert.equal(c.continuation_prompt, "A new scene.");
  assert.equal(duration.value, 10);
});

test("legacy upload keeps the upload source and fractional duration", () => {
  const duration = { value: 5 };
  const c = normalizeContinuity({ version: 2, operation: "continue", source_kind: "video",
    source_video_id: "upload", extension_frames: 136, continuation_prompt: "Move ahead." }, duration);
  assert.equal(c.operation, "continue");
  assert.equal(c.source_video_id, "upload");
  assert.equal(duration.value, 136 / 24);
});

test("legacy stock prefill becomes automatic policy without losing the idea", () => {
  const prefill = "Continue the same uninterrupted shot naturally. Preserve the subjects' identity, clothing, positions, lighting and environment. Maintain the established motion direction, camera trajectory and ambient sound. Do not restart the action, repeat completed dialogue, introduce a cut, fade, title, freeze or loop.";
  const c = normalizeContinuity({ operation: "continue", source_id: "parent", extension_frames: 119,
    continuation_prompt: prefill, idea: "Speed up after the seam." }, { value: 5 });
  assert.equal(c.version, 3);
  assert.equal(c.continuation_prompt, "Speed up after the seam.");
});

test("v3 keeps its prompt, Duration, saved Forge state and source activation", () => {
  const duration = { value: 10 };
  const snapshot = { "<Picture 1>": "same-reference" };
  const c = normalizeContinuity({ version: 3, source_id: "parent", operation: "new",
    extension_frames: 119, continuation_prompt: "", forge_reference_snapshot: snapshot }, duration);
  assert.equal(c.operation, "continue");
  assert.equal(duration.value, 10);
  assert.equal(c.continuation_prompt, "");
  assert.equal(c.forge_reference_snapshot, snapshot);
  c.source_id = "";
  assert.equal(normalizeContinuity(c, duration).operation, "new");
});


for (const frames of [17, 357]) test(`legacy boundary ${frames} frames survives round trip`, () => {
  const duration = { value: 5 };
  let c = normalizeContinuity({ version: 2, operation: "continue", source_id: "parent",
    extension_frames: frames, continuation_prompt: "", capture: true }, duration);
  assert.equal(duration.value, frames / 24);
  assert.equal(continuityTiming(duration.value).extension_frames, frames);
  c = JSON.parse(JSON.stringify(c));
  normalizeContinuity(c, duration);
  assert.equal(duration.value, frames / 24);
  assert.equal(c.capture, true);
});

test("short source shrinks overlap without truncating new frames", () => {
  assert.equal(continuityTiming(5, 73, 5).overlap_frames, 5);
  assert.equal(continuityTiming(5, 73, 21).overlap_frames, 5);
  assert.equal(continuityTiming(5, 73, 22).overlap_frames, 22);
  assert.equal(continuityTiming(15, 73, 124).overlap_frames, 5);
  assert.equal(continuityTiming(15, 73, 124).extension_frames, 357);
  assert.throws(() => continuityTiming(5, 22, 4), /at least 5/);
});

test("timing rejects invalid Duration values", () => {
  for (const seconds of [0, -1, 15.01, NaN, Infinity]) {
    assert.throws(() => continuityTiming(seconds), /Duration/);
  }
});
