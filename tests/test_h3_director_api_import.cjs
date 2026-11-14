const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = fs.readFileSync(path.resolve(__dirname, '../js/minimax_h3_director.js'), 'utf8');
const inputs = require('./fixtures/h3_director_issue_100_inputs.json');
function region(start, end) {
  assert.ok(source.includes(start) && source.includes(end));
  return source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
}
function harness({ nested = false } = {}) {
  const frames = [];
  const node = { comfyClass: 'MiniMaxH3Director', widgets: Object.entries({ mode: 'FL2VA', prompt: '', width: 1344, height: 768, duration: 5, timeline_data: '{}', builder_state: '', frame_rate: 24 }).map(([name, value]) => ({ name, value })) };
  const widget = name => node.widgets.find(w => w.name === name);
  const app = { graph: { _nodes: [nested ? { subgraph: { _nodes: [node] } } : node] }, registerExtension(ext) { this.extension = ext; } };
  const context = vm.createContext({ app, node, structuredClone, requestAnimationFrame: fn => frames.push(fn),
    install() {}, render() {}, syncNodeBounds() {}, refreshContinuity() {}, continuityState() {},
    hasExternalCanvas: () => false, activeItems: () => context.savedState().items,
  });
  vm.runInContext('let loadingGraph = false;\n' + region('const DEFAULT_BUILDER_STATE =', 'const MAX =') + '\nconst MINIMAX_MULTIPLE = 32; const RESOLUTION_PRESETS = {};\n' +
    region('function parseState(', 'function viewUrl(') + '\n' +
    `const mode = () => node.widgets.find(w => w.name === 'mode').value;
     const dataWidget = node.widgets.find(w => w.name === 'timeline_data');
     const builderWidget = node.widgets.find(w => w.name === 'builder_state');
     const promptWidget = node.widgets.find(w => w.name === 'prompt');
     const widthWidget = node.widgets.find(w => w.name === 'width');
     const heightWidget = node.widgets.find(w => w.name === 'height');
     const modeWidget = null; const externalPromptWidget = () => null;
     let state = parseState('{}'), builderState = { ...DEFAULT_BUILDER_STATE(mode()), simple_prompt: '', prompt_mode: 'simple' }, selectedId = null;
     const fieldHeights = {};
     globalThis.savedState = () => state;
    ` +
    region('  function migratePromptToSingleField()', '  const refModLibrary =') + '\n' +
    region('  function builderPromptForWidget(', '  function previewTextFor(') + '\n' +
    region('  const resolutionState =', '  const fileInput =') + '\n' +
    region('  const restorePersistedState =', '  const oldConnectionsChange =') + '\n' +
    region('  node.__dasiwaH3RestorePersistedState =', '  node.__dasiwaH3State =') + '\n' +
    region('  if (modeWidget) { const old = modeWidget.callback;', '  const oldDrawForeground =') + '\n' +
    source.slice(source.indexOf('app.registerExtension({')),
    context);
  return { node, widget, app, context, frames, load(values) {
    app.extension.beforeLoadGraph?.();
    // Frontend 1.59.1 loadApiJsonNow assigns and invokes callbacks in input order.
    for (const [name, value] of Object.entries(values)) {
      const w = widget(name); if (w) { w.value = value; w.callback?.(value); }
    }
    app.extension.afterConfigureGraph?.();
  } };
}
test('fresh nodes still emit defaults and manual clearing stays empty', () => {
  const h = harness();
  h.widget('duration').callback(5);
  assert.equal(h.widget('prompt').value, '');
  assert.equal(JSON.parse(h.widget('builder_state').value).duration, 5);
  h.load(inputs);
  vm.runInContext("builderState.simple_prompt = ''; emit();", h.context);
  h.widget('duration').callback(5);
  assert.equal(h.widget('prompt').value, '');
  assert.equal(JSON.parse(h.widget('builder_state').value).simple_prompt, '');
});
for (const hook of ['onGraphLoadError', 'afterLoadGraph']) {
  test(`${hook} releases import suppression`, () => {
    const h = harness();
    h.app.extension.beforeLoadGraph();
    h.widget('duration').value = 7;
    h.widget('duration').callback(7);
    assert.equal(h.widget('builder_state').value, '');
    h.app.extension[hook](new Error('cancelled'));
    h.widget('duration').callback(7);
    assert.equal(JSON.parse(h.widget('builder_state').value).duration, 7);
  });
}
test('Director inside a subgraph is restored after graph load', () => {
  const h = harness({ nested: true });
  h.load(inputs);
  assert.equal(h.widget('prompt').value, inputs.prompt);
  assert.equal(h.context.savedState().items.length, 2);
});
for (const emptyBuilder of [false, true]) {
  test(`API import restores complete state before workflow snapshot (empty builder=${emptyBuilder})`, () => {
    const h = harness();
    const timeline = JSON.parse(inputs.timeline_data);
    if (emptyBuilder) delete timeline.builder_state;
    h.load({ ...inputs, timeline_data: JSON.stringify(timeline), ...(emptyBuilder ? { builder_state: '' } : {}) });
    assert.equal(h.widget('prompt').value, inputs.prompt);
    assert.equal(h.context.savedState().items.length, 2);
    assert.deepEqual(JSON.parse(h.widget('timeline_data').value).items, JSON.parse(inputs.timeline_data).items);
    assert.equal(JSON.parse(h.widget('builder_state').value).simple_prompt, inputs.prompt);
    const saved = { widgets_values: h.node.widgets.map(w => w.value) };
    h.node.onSerialize(saved);
    assert.deepEqual([h.widget('width').value, h.widget('height').value], [768, 1152]);
    assert.equal(saved.widgets_values[1], inputs.prompt);
    // Visual workflow reload and duration/FPS edits must not restore defaults.
    h.node.onConfigure({});
    while (h.frames.length) h.frames.shift()();
    h.widget('duration').value = 6; h.widget('duration').callback(6);
    h.widget('frame_rate').callback(24);
    assert.equal(h.widget('prompt').value, inputs.prompt);
    assert.equal(h.context.savedState().items.length, 2);
    assert.deepEqual([h.widget('width').value, h.widget('height').value], [768, 1152]);
  });
}
