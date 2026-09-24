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
console.log('H3 prompt editor migrations passed');
