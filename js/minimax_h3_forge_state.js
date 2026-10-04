// Serialized reference intent and prompt fields shared by Director and Forge.
const REF_KEYS = ["subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"];
const IMAGE_ROLES = ["subject", "style", "keyframe", "pose", "custom"];

export function refPromptFields(text) {
  const value = String(text || "").trim();
  const matches = [...value.matchAll(/^[ \t]*(subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music|soundscape|music):[ \t]*/gmi)];
  const keys = matches.map(m => m[1].toLowerCase()).map(key => key === "overall_soundscape" ? "soundscape" : key === "non_diegetic_music" ? "music" : key);
  if (matches.length !== REF_KEYS.length || matches[0].index !== 0 || new Set(keys).size !== REF_KEYS.length) return null;
  return Object.fromEntries(matches.map((m, i) => [keys[i], value.slice(m.index + m[0].length, matches[i + 1]?.index ?? value.length).trim()]));
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

// REF2VA picture labels (item.forge_label): what each picture is, picked in
// Forge. "character-2" is Character 2, "group-21" a picture with Characters
// 2 and 1 in it (2 on the left), and the rest name themselves. Pictures with
// the same Character number are one subject, which is what a subject group
// was, so a label also sets the role and group the other paths read.
export const PICTURE_LABELS = [...Array.from({ length: 32 }, (_, i) => `character-${i + 1}`), "group-12", "group-21", "group-13", "group-31",
  "group-23", "group-32", "group-123", "place", "style", "first-frame", "last-frame", "pose", "custom"];

export function labelRole(label) {
  if (label.startsWith("character-")) return { forge_role: "subject", forge_subject_group: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[Number(label.slice(10)) - 1] || label };
  const role = { style: "style", "first-frame": "keyframe", "last-frame": "keyframe", pose: "pose", custom: "custom" }[label] || "subject";
  return { forge_role: role, forge_subject_group: "" };
}

// A picture never labelled gets one from its role and group: a group becomes
// one Character number, and an ungrouped subject the next free one, which is
// what "Separate" meant.
function pictureLabels(images) {
  const saved = new Map(images.filter(i => PICTURE_LABELS.includes(i.forge_label)).map(i => [i.id, i.forge_label]));
  const used = new Set([...saved.values()].flatMap(v => v.startsWith("character-") ? [Number(v.slice(10))] : v.startsWith("group-") ? [...v.slice(6)].map(Number) : []));
  const take = () => { let n = 1; while (used.has(n)) n += 1; used.add(n); return n; };
  const byGroup = new Map(images.filter(i => i.group && saved.get(i.id)?.startsWith("character-")).map(i => [i.group, Number(saved.get(i.id).slice(10))]));
  return Object.fromEntries(images.map(item => {
    if (saved.has(item.id)) return [item.id, saved.get(item.id)];
    const role = IMAGE_ROLES.includes(item.forge_role) ? item.forge_role : "subject";
    if (role !== "subject") return [item.id, { style: "style", keyframe: "first-frame", pose: "pose", custom: "custom" }[role]];
    const group = item.group;
    if (group && !byGroup.has(group)) byGroup.set(group, take());
    return [item.id, `character-${group ? byGroup.get(group) : take()}`];
  }));
}

export function forgeReferences(items, mode, legacyGroups = {}) {
  const order = { image: 0, video: 1, audio: 2 };
  const sorted = items.filter(item => item.enabled !== false && item.value != null && order[item.type] !== undefined)
    .slice().sort((a, b) => order[a.type] - order[b.type] || Number(a.slot ?? 0) - Number(b.slot ?? 0) || Number(a.order ?? 0) - Number(b.order ?? 0));
  const labels = mode === "REF2VA" ? pictureLabels(sorted.filter(item => item.type === "image").map(item => ({ ...item, group: String(item.forge_subject_group ?? legacyGroups[item.id] ?? "") }))) : {};
  return sorted
    .map(item => {
      const common = { item, instructions: String(item.forge_instructions || ""), keep: String(item.forge_keep || ""), drop: String(item.forge_drop || "") };
      if (item.type === "image") {
        const role = mode === "REF2VA" && IMAGE_ROLES.includes(item.forge_role) ? item.forge_role : mode === "REF2VA" ? "subject" : "keyframe";
        return { ...common, kind: "image", path: typeof item.value === "string" ? item.value : undefined, role, subject_group: role === "subject" ? String(item.forge_subject_group ?? legacyGroups[item.id] ?? "") : "", ...(labels[item.id] ? { easy_role: labels[item.id] } : {}) };
      }
      if (item.type === "audio") return { ...common, kind: "audio", duration_seconds: item.duration };
      return { ...common, kind: "video", role: "motion", stream: item.media_mode === "audio" ? "audio" : item.media_mode === "video_audio" || item.audio != null ? "both" : "video", duration_seconds: item.duration };
    });
}
