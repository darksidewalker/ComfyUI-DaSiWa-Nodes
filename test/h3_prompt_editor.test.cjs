// Exercise the Director's real frontend migration/template functions without ComfyUI.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('js/minimax_h3_director.js', 'utf8');
const between = (start, end) => source.slice(source.indexOf(start), source.indexOf(end));
const code = [
  between('const DEFAULT_BUILDER_STATE =', 'const DEFAULT_STATE ='),
  between('function textValue(', 'function viewUrl('),
  between('function insertAtCursor(', 'function createBuilderField('),
  between('  function migratePromptToSingleField() {', '  const emit ='),
  between('  function builderPromptForWidget(', '  function previewTextFor('),
  between('  const joinText =', '  // Old reference packs'),
  between('  function portablePromptText(', '  // Places incoming items'),
  between('  function refPrefill(', '  function refModTagMap()'),
  'this.DEFAULT_BUILDER_STATE = DEFAULT_BUILDER_STATE;',
].join('\n');
function evaluate(builder, timeline = {}, widget = '', currentMode = 'REF2VA') {
  const context = { builderState: builder, state: timeline, promptWidget: { value: widget }, mode: () => currentMode, Event: class { constructor(type) { this.type = type; } } };
  vm.runInNewContext(code.replace('  migratePromptToSingleField();', ''), context);
  context.migratePromptToSingleField();
  return context;
}
const blank = evaluate({}, {}, '', 'T2VA');
assert.equal(blank.builderState.simple_prompt, '');
assert.equal(blank.builderState.prompt_mode, 'simple');
assert.match(blank.builderPromptForWidget(blank.DEFAULT_BUILDER_STATE('FL2VA'), 'FL2VA'), /integrated_multimodal_description:/);
const old = evaluate({ prompt_mode: 'structured', simple_prompt: 'stale', ref: { detailed_description: 'current shot' } });
assert.match(old.builderState.simple_prompt, /current shot/);
assert.doesNotMatch(old.builderState.simple_prompt, /stale/);
const oldVideo = evaluate({ prompt_mode: 'structured', ref: { subject_defs: [{ text: 'red lion' }] } });
assert.match(oldVideo.builderState.simple_prompt, /red lion/);
const embedded = evaluate({}, { resolved_prompt: 'restored from video metadata' });
assert.equal(embedded.builderState.simple_prompt, 'restored from video metadata');
const explicit = evaluate({ prompt_mode: 'simple', simple_prompt: '' }, { resolved_prompt: 'obsolete' });
assert.equal(explicit.builderState.simple_prompt, '');
const pack = evaluate({}, {}, '', 'REF2VA');
pack.appendPortablePrompt({ prompt_mode: 'structured', fields: { subject_definitions: 'fox', detailed_description: '[Shot 1] runs' } });
assert.match(pack.builderState.simple_prompt, /fox/);
assert.match(pack.builderState.simple_prompt, /\[Shot 1\] runs/);
pack.overwritePortablePrompt({ prompt_mode: 'structured', fields: { subject_definitions: 'new fox' } });
assert.match(pack.builderState.simple_prompt, /new fox/);
assert.doesNotMatch(pack.builderState.simple_prompt, /\[Shot 1\] runs/);
const events = [];
const area = { value: 'begin end', selectionStart: 6, selectionEnd: 9, focus() {}, dispatchEvent(event) { events.push(event.type); } };
pack.insertAtCursor(area, '[Shot 1]');
assert.equal(area.value, 'begin [Shot 1]');
assert.deepEqual(events, ['input', 'change']);
const prefill = pack.refPrefill([
  { type: 'image', slot: 1, prompt: 'blue coat' },
  { type: 'image', slot: 0, enabled: false },
  { type: 'video', slot: 0, media_mode: 'video_audio', audioSlot: 2, prompt: 'camera circles' },
  { type: 'video', slot: 1, media_mode: 'audio', prompt: 'violin rhythm' },
  { type: 'audio', slot: 0, prompt: 'low strings' },
], [{ name: 'combo', slot: 1, description: 'saved face' }], [{ name: 'combo', kinds: ['image', 'audio'] }]);
assert.match(prefill.subject_definitions, /<Picture 1>.*blue coat/);
assert.match(prefill.subject_definitions, /<Picture 2>.*saved face/);
assert.match(prefill.subject_definitions, /<Video 1>.*camera circles/);
assert.doesNotMatch(prefill.subject_definitions, /<Video 2>/);
assert.match(prefill.subject_definitions, /<Audio 1>.*synchronized/);
assert.match(prefill.subject_definitions, /<Audio 2>.*violin rhythm/);
assert.match(prefill.subject_definitions, /<Audio 3>.*low strings/);
assert.match(prefill.subject_definitions, /<Audio 4>.*saved face/);
assert.match(prefill.summary, /^\[reference generation \+ audio reference\]/);
assert.doesNotMatch(prefill.summary, /video editing/);
assert.match(prefill.retention_analysis, /<Audio 1>: reference/);
const filled = pack.applyRefPrefill('', prefill);
assert.match(filled, /subject_definitions:\n<Picture 1>/);
assert.match(filled, /retention_analysis:\n<Picture 1>/);
assert.match(filled, /detailed_description:\n\n/);
assert.match(filled, /non_diegetic_music:\nN\/A/);
assert.equal(pack.applyRefPrefill(filled, prefill), filled);
const authored = pack.applyRefPrefill('subject_definitions:\nmy subject\n\nsummary:\n\nretention_analysis:\nmy own analysis\n\ndetailed_description:\n[Shot 1] moves', prefill);
assert.match(authored, /subject_definitions:\nmy subject/);
assert.match(authored, /summary:\n\[reference generation/);
assert.match(authored, /retention_analysis:\nmy own analysis/);
assert.match(authored, /\[Shot 1\] moves/);
assert.match(pack.applyRefPrefill('Already written scene', prefill), /detailed_description:\nAlready written scene/);
assert.equal(pack.refPrefill([{ type: 'video', slot: 0, media_mode: 'audio' }], [], []).summary.startsWith('[audio reference]'), true);
assert.equal(pack.refPrefill([], [], []), null);
assert.doesNotMatch(source.slice(source.indexOf('  function buildSimpleForm('), source.indexOf('  function refPrefill(')), /Preview Prompt|window\.prompt\(/);
const popoverCode = between('  let closePromptNumberPopover = null;', '  function addCharCounter(');
const elements = [];
const listeners = {};
function element() {
  const e = { style: {}, children: [], offsetWidth: 194, offsetHeight: 85,
    append(...children) { this.children.push(...children); },
    addEventListener(type, fn) { this[`on_${type}`] = fn; },
    contains(target) { return this === target || this.children.includes(target); },
    remove() { this.removed = true; }, focus() {}, select() {}, setCustomValidity() {}, reportValidity() {} };
  elements.push(e); return e;
}
const pop = { document: { createElement: element, body: { append() {} },
  addEventListener(type, fn) { listeners[type] = fn; }, removeEventListener(type) { delete listeners[type]; } },
  window: { innerWidth: 800, innerHeight: 600, addEventListener(type, fn) { listeners[type] = fn; }, removeEventListener(type) { delete listeners[type]; } } };
vm.runInNewContext(popoverCode, pop);
let number = null;
const anchor = { getBoundingClientRect() { return { left: 30, top: 40, bottom: 60 }; }, focus() {} };
pop.openPromptNumberPopover(anchor, 'Shot number', n => { number = n; });
let box = elements[0];
assert.equal(box.className, 'ds-h3-number-popover');
assert.equal(box.style.left, '30px');
box.children[1].value = '3';
box.children[2].onclick();
assert.equal(number, 3);
assert.equal(box.removed, true);
pop.openPromptNumberPopover(anchor, 'RefMod number', n => { number = n; });
box = elements[4];
listeners.keydown({ key: 'Escape', stopPropagation() {} });
assert.equal(box.removed, true);
assert.equal(number, 3);
console.log('H3 prompt editor migrations, prefill and number popovers passed');
