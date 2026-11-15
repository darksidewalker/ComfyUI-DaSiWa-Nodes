const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = fs.readFileSync(path.resolve(__dirname, '../js/minimax_h3_director.js'), 'utf8');
function region(start, end) {
  const from = source.indexOf(start), to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from);
  return source.slice(from, to);
}
function harness() {
  const context = vm.createContext({});
  vm.runInContext(region('const MINIMAX_MULTIPLE =', 'const REPOSITORY_URL =') + `
    let state = { resolution: null };
    let loadingGraph = false;
    const node = { widgets: ['width', 'height'].map(name => ({ name, value: 0 })) };
    const activeItems = () => [];
    const hasExternalCanvas = () => false;
  ` + region('  const resolutionState =', '  const fileInput =') + `
    globalThis.choices = ASPECT_OPTIONS;
    globalThis.apply = settings => {
      state.resolution = settings;
      applyResolution();
      return node.widgets.map(w => w.value);
    };
  `, context);
  return context;
}
for (const [aspect, expected] of [['7:9', [896, 1152]], ['9:7', [1152, 896]]]) {
  test(`Director offers ${aspect} and writes dimensions on its existing 32-pixel grid`, () => {
    const h = harness();
    assert.ok(h.choices.some(([id, label]) => id === aspect && label === aspect));
    assert.deepEqual(Array.from(h.apply({ aspect, resolution: '1024p' })), expected);
    assert.deepEqual(Array.from(h.apply({ aspect, resolution: 'custom', custom_mp: 1 })), expected);
    assert.deepEqual(Array.from(h.apply({ aspect, resolution: 'auto' })), aspect === '7:9' ? [768, 992] : [992, 768]);
  });
}
