// Pure frontend counterparts of nodes/h3_continuity/core.py. Tested for parity.
export const CONTINUITY_DEFAULTS = { version: 3, source_kind: "checkpoint", source_id: "", source_video_id: "", source_video: null, use_references: false, operation: "new", capture: false, session: "", overlap_frames: 22, continuation_prompt: "", idea: "" };
export const CONTINUITY_LEGACY_PROMPT = "Continue the same uninterrupted shot naturally. Preserve the subjects' identity, clothing, positions, lighting and environment. Maintain the established motion direction, camera trajectory and ambient sound. Do not restart the action, repeat completed dialogue, introduce a cut, fade, title, freeze or loop.";
export const sourceId = c => c?.source_kind === "video" ? c.source_video_id : c?.source_id;
export function continuityTiming(seconds, preferred = 22, sourceFrames = null) {
  seconds = Number(seconds);
  if (!Number.isFinite(seconds) || seconds <= 0 || seconds > 15) throw new Error("Set Duration above 0 and at most 15 seconds for Continuity.");
  const extension = Math.max(17, Math.floor(seconds * 24 / 17 + .5) * 17);
  const available = Math.min(preferred, 362 - extension, sourceFrames ?? Infinity);
  const overlap = [73, 56, 39, 22, 5].find(n => n <= available);
  if (overlap == null) throw new Error("The source needs at least 5 H3 frames.");
  return { duration_seconds: seconds, overlap_frames: overlap, extension_frames: extension, added_seconds: extension / 24, window_frames: overlap + extension };
}
export function migrateContinuity(c, durationWidget) {
  if (!c || typeof c !== "object") return { ...CONTINUITY_DEFAULTS };
  if (Number(c.version || 2) < 3) {
    if (c.operation === "continue" && sourceId(c)) {
      if (durationWidget && Number(c.extension_frames) > 0) durationWidget.value = Number(c.extension_frames) / 24;
    } else {
      // Old UI allowed a preselected, inactive source. Do not start it on upgrade.
      c.source_id = ""; c.source_video_id = "";
    }
    if (c.continuation_prompt === CONTINUITY_LEGACY_PROMPT) c.continuation_prompt = "";
    if (c.idea?.trim()) c.continuation_prompt = [c.continuation_prompt, c.idea.trim()].filter(Boolean).join("\n");
    c.idea = ""; c.version = 3;
  }
  for (const [key, value] of Object.entries(CONTINUITY_DEFAULTS)) if (c[key] === undefined) c[key] = value;
  // The old independent control must never survive as a second timing authority.
  delete c.extension_frames; delete c.window_frames; delete c.added_seconds; delete c.duration_seconds;
  c.operation = sourceId(c) ? "continue" : "new";
  return c;
}

export function newContinuitySession() {
  // getRandomValues also works on plain-HTTP LAN ComfyUI; randomUUID does not.
  return "h3_" + Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, "0")).join("");
}
