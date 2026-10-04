// Exercise Forge's real reference mapping without loading the ComfyUI browser app.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const stateSource = readFileSync(new URL("../../js/minimax_h3_forge_state.js", import.meta.url), "utf8");
const { forgeReferences, labelRole } = await import("data:text/javascript;base64," + Buffer.from(stateSource).toString("base64"));

const image = (id, slot, extra = {}) => ({ id, type: "image", slot, value: `${id}.png`, ...extra });

const refs = forgeReferences([image("p1", 0), image("p2", 1, { forge_role: "keyframe" }), image("p3", 2)], "REF2VA",
  { p1: "A", p2: "A", p3: "A" });
assert.equal(refs.length, 3);
assert.deepEqual(refs.map(r => r.subject_group), ["A", "", "A"]);
assert.deepEqual(refs.map(r => r.role), ["subject", "keyframe", "subject"]);
const base = forgeReferences([image("p1", 0)], "I2VA", { p1: "A" });
assert.equal(base[0].subject_group, "");
assert.equal(base[0].easy_role, undefined);

// A saved label wins; an unlabelled picture gets the next free Character
// number, and its role says what else it is.
const labels = forgeReferences([image("p1", 0), image("p2", 1, { forge_label: "place" }), image("p3", 2, { forge_label: "character-1" }),
  image("p4", 3, { forge_role: "pose" })], "REF2VA");
assert.deepEqual(labels.map(r => r.easy_role), ["character-2", "place", "character-1", "pose"]);
// A node saved with subject groups: a group becomes one Character, each
// ungrouped picture its own.
const carried = forgeReferences([image("p1", 0), image("p2", 1), image("p3", 2)], "REF2VA", { p1: "A", p3: "A" });
assert.deepEqual(carried.map(r => r.easy_role), ["character-1", "character-2", "character-1"]);

// A label also sets the role and group that continuity and the Director read.
assert.deepEqual(labelRole("character-2"), { forge_role: "subject", forge_subject_group: "B" });
assert.deepEqual(labelRole("group-21"), { forge_role: "subject", forge_subject_group: "" });
assert.deepEqual(labelRole("last-frame"), { forge_role: "keyframe", forge_subject_group: "" });
assert.deepEqual(labelRole("custom"), { forge_role: "custom", forge_subject_group: "" });
// Separate legacy references must not silently merge after Character 4.
const many = forgeReferences(Array.from({ length: 9 }, (_, i) => image(`many-${i}`, i)), "REF2VA");
assert.equal(new Set(many.map(r => r.easy_role)).size, 9);
assert.equal(many[8].easy_role, "character-9");
const partial = forgeReferences([image("p1", 0, { forge_label: "character-1", forge_subject_group: "A" }), image("p2", 1, { forge_subject_group: "A" })], "REF2VA");
assert.deepEqual(partial.map(r => r.easy_role), ["character-1", "character-1"]);
assert.deepEqual(labelRole("character-9"), { forge_role: "subject", forge_subject_group: "I" });
console.log("Forge reference mapping: PASS");
