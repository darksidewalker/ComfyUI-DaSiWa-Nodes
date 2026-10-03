import { refPromptFields } from "./minimax_h3_forge_state.js";

// Pure frontend counterparts of nodes/h3_continuity/core.py. Tested for parity.
export const CONTINUITY_DEFAULTS = { version: 3, source_kind: "checkpoint", source_id: "", source_video_id: "", source_video: null, use_references: false, operation: "new", capture: false, session: "", overlap_frames: 22, continuation_prompt: "", idea: "" };
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
export function normalizeContinuity(c, durationWidget) {
  if (!c || typeof c !== "object") return { ...CONTINUITY_DEFAULTS };
  if (Number(c.version || 2) < 3) {
    if (c.operation === "continue" && sourceId(c)) {
      // Migrate the old +frames authority once, before removing derived fields.
      const frames = Number(c.extension_frames);
      if (durationWidget && Number.isFinite(frames) && frames > 0) durationWidget.value = frames / 24;
    } else {
      // A legacy preselected source could be inactive. Preserve that choice.
      c.source_id = ""; c.source_video_id = ""; c.source_video = null;
    }
    const legacyPrefill = "Continue the same uninterrupted shot naturally. Preserve the subjects' identity, clothing, positions, lighting and environment. Maintain the established motion direction, camera trajectory and ambient sound. Do not restart the action, repeat completed dialogue, introduce a cut, fade, title, freeze or loop.";
    if (c.continuation_prompt === legacyPrefill) c.continuation_prompt = "";
    if (c.idea?.trim()) {
      const idea = c.idea.trim(), fields = refPromptFields(c.continuation_prompt);
      if (fields) {
        // Unlabelled trailing prose would become part of the final (music) field.
        fields.detailed_description = [fields.detailed_description, `Next action: ${idea}`].filter(Boolean).join("\n");
        c.continuation_prompt = Object.entries(fields).map(([key, value]) =>
          `${key === "soundscape" ? "overall_soundscape" : key === "music" ? "non_diegetic_music" : key}:\n${value}`).join("\n\n");
      } else c.continuation_prompt = [c.continuation_prompt, idea].filter(Boolean).join("\n");
    }
    c.idea = "";
    c.version = 3;
  }
  for (const [key, value] of Object.entries(CONTINUITY_DEFAULTS)) if (c[key] === undefined) c[key] = value;
  // Duration is the sole timing authority; remove any stale derived fields.
  delete c.extension_frames; delete c.window_frames; delete c.added_seconds; delete c.duration_seconds;
  c.operation = sourceId(c) ? "continue" : "new";
  return c;
}

export function newContinuitySession() {
  // getRandomValues also works on plain-HTTP LAN ComfyUI; randomUUID does not.
  return "h3_" + Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, "0")).join("");
}
