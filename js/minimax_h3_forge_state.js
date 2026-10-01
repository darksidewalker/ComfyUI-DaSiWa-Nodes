// Serialized reference intent and prompt fields shared by Director and Forge.
const REF_KEYS = ["subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"];
const IMAGE_ROLES = ["subject", "style", "keyframe", "pose", "custom"];

export function refPromptFields(text) {
  const value = String(text || "").trim();
  const matches = [...value.matchAll(/^[ \t]*(subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music|soundscape|music):[ \t]*/gm)];
  if (matches.length !== REF_KEYS.length || matches[0].index !== 0 || new Set(matches.map(m => m[1] === "soundscape" ? "overall_soundscape" : m[1] === "music" ? "non_diegetic_music" : m[1])).size !== REF_KEYS.length) return null;
  return Object.fromEntries(matches.map((m, i) => [m[1] === "overall_soundscape" ? "soundscape" : m[1] === "non_diegetic_music" ? "music" : m[1], value.slice(m.index + m[0].length, matches[i + 1]?.index ?? value.length).trim()]));
}

export function refTemplate(text = "", definitions) {
  const fields = refPromptFields(text);
  if (fields && definitions === undefined) return text;
  return REF_KEYS.map(key => `${key}:\n${key === "subject_definitions" ? definitions || "" : fields ? fields[key === "overall_soundscape" ? "soundscape" : key === "non_diegetic_music" ? "music" : key] : key === "detailed_description" ? String(text).trim() : ""}`).join("\n\n");
}

export function inheritedDefinitions(text, previous = {}, current = {}) {
  let removed = false;
  const byIdentity = Object.fromEntries(Object.entries(current).map(([tag, identity]) => [identity, tag]));
  const result = String(text || "").replace(/<(Picture|Video|Audio)\s+(\d+)>/g, (tag, kind, n) => {
    const identity = previous[`<${kind} ${Number(n)}>`];
    const replacement = identity && byIdentity[identity];
    if (replacement) return replacement;
    removed = true;
    return "";
  });
  return { text: result.trim(), warning: removed ? "Existing identities were kept; unverified media links were removed. Forge will relate them to the current references." : "" };
}

export function referenceTags(references) {
  // Native conditioning places paired video audio before standalone audio.
  const counts = { Picture: 0, Video: 0, Audio: references.filter(r => r.kind === "video" && r.stream === "both" && !r.saved_reference).length };
  let pairedAudio = 0;
  return references.map(ref => {
    const kinds = ref.kind === "image" ? ["Picture"] : ref.kind === "audio" ? ["Audio"] : ref.stream === "audio" ? ["Audio"] : ref.stream === "both" ? ["Video", "Audio"] : ["Video"];
    return kinds.map(kind => `<${kind} ${kind === "Audio" && ref.kind === "video" && ref.stream === "both" && !ref.saved_reference ? ++pairedAudio : ++counts[kind]}>`);
  });
}

export function referenceSnapshot(references) {
  const result = {}, tags = referenceTags(references);
  references.forEach((ref, index) => {
    for (const tag of tags[index]) {
      const kind = tag.slice(1).split(" ")[0];
      result[tag] = `${ref.item?.id || ref.id || ""}:${typeof ref.item?.value === "string" ? ref.item.value : ref.path || ""}:${ref.item?.trim_start ?? 0}:${ref.item?.trim_end ?? ""}:${kind}`;
    }
  });
  return result;
}

export function forgeReferences(items, mode, legacyGroups = {}) {
  const order = { image: 0, video: 1, audio: 2 };
  return items.filter(item => item.enabled !== false && item.value != null && order[item.type] !== undefined)
    .slice().sort((a, b) => order[a.type] - order[b.type] || Number(a.slot ?? 0) - Number(b.slot ?? 0) || Number(a.order ?? 0) - Number(b.order ?? 0))
    .map(item => {
      const common = { item, instructions: String(item.forge_instructions || ""), keep: String(item.forge_keep || ""), drop: String(item.forge_drop || "") };
      if (item.type === "image") {
        const role = mode === "REF2VA" && IMAGE_ROLES.includes(item.forge_role) ? item.forge_role : mode === "REF2VA" ? "subject" : "keyframe";
        return { ...common, kind: "image", path: typeof item.value === "string" ? item.value : undefined, role, subject_group: role === "subject" ? String(item.forge_subject_group ?? legacyGroups[item.id] ?? "") : "" };
      }
      if (item.type === "audio") return { ...common, kind: "audio", duration_seconds: item.duration };
      return { ...common, kind: "video", role: "motion", stream: item.media_mode === "audio" ? "audio" : item.media_mode === "video_audio" || item.audio != null ? "both" : "video", duration_seconds: item.duration };
    });
}
