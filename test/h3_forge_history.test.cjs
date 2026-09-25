// Run Forge's real frontend in a small DOM/API harness; no model or browser needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('js/minimax_h3_forge.js', 'utf8').replace(/^import .*;\n/gm, '');
const all = [];
const listeners = new Map();
function makeElement(tag) {
  const element = { tag, children: [], style: {}, hidden: false,
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    insertBefore(child) { this.children.push(child); },
    addEventListener(type, fn) { this[`on_${type}`] = fn; },
    remove() { this.removed = true; },
    focus() {},
    classList: { toggle() {} },
  };
  all.push(element);
  return element;
}
const document = {
  createElement: makeElement, getElementById() { return null; },
  head: { append() {} }, body: { append() {} },
  addEventListener(type, fn) { listeners.set(type, fn); },
  removeEventListener(type) { listeners.delete(type); },
};
const storage = new Map();
const localStorage = { getItem(k) { return storage.get(k); }, setItem(k, v) { storage.set(k, v); } };
let generations = 0;
const api = {
  fetchApi: async (path) => {
    if (path.endsWith('/models')) return { ok: true, json: async () => ({ models: [{ id: 'local:test', label: 'Test model' }], creativity: ['balanced'], default_creativity: 'balanced', detail_levels: { 5: 'Standard' }, default_detail: 5, errors: {} }) };
    if (path.endsWith('/cancel')) return { ok: true };
    generations++;
    return { ok: true, json: async () => ({ simple_prompt: `draft ${generations}`, fields: { imd: `draft ${generations}` }, mode: 'T2VA', model: 'local:test', stats: { seconds: 0, output_tokens: 1 }, warnings: [], unloaded: true }) };
  },
};
const context = { document, localStorage, api, app: { registerExtension() {} }, window: {},
  setInterval: () => 1, clearInterval() {}, Date, Math, JSON, console };
vm.runInNewContext(source, context);
const applied = [];
const node = { id: 7, graph: { setDirtyCanvas() {} }, __dasiwaH3Forge: {
  mode: () => 'T2VA', duration: () => 5, items: () => [], setStatus() {}, apply: result => applied.push(result.simple_prompt),
} };
const elements = (tag, name) => all.filter(e => e.tag === tag && e.textContent === name);
async function openAndGenerate() {
  await context.window.DaSiWaH3Forge.open(node);
  const box = all.filter(e => e.className === 'ds-forge').at(-1);
  const brief = box.children.find(e => e.className === 'field').children[1];
  brief.value = 'a scene';
  const generate = elements('button', 'Generate').at(-1);
  await generate.onclick();
  return box;
}
(async () => {
  for (let i = 0; i < 4; i++) {
    const box = await openAndGenerate();
    box.children[0].children[1].onclick(); // close without applying
  }
  assert.deepEqual(Array.from(node.properties.dasiwaH3ForgeHistory, x => x.simple_prompt), ['draft 4', 'draft 3', 'draft 2']);
  assert.equal(applied.length, 0);
  await context.window.DaSiWaH3Forge.open(node);
  const box = all.filter(e => e.className === 'ds-forge').at(-1);
  const history = box.children.find(e => e.className === 'history');
  assert.equal(history.children.length, 4); // heading + three choices
  const preview = box.children.find(e => e.tag === 'pre');
  assert.equal(preview.textContent, 'draft 4');
  history.children[2].onclick();
  assert.equal(preview.textContent, 'draft 3');
  elements('button', 'Apply to node').at(-1).onclick();
  assert.deepEqual(applied, ['draft 3']);
  // Serialized node properties retain drafts across workflow reloads, not just dialog reopen.
  const restored = { id: 8, properties: JSON.parse(JSON.stringify(node.properties)), __dasiwaH3Forge: node.__dasiwaH3Forge };
  await context.window.DaSiWaH3Forge.open(restored);
  const restoredBox = all.filter(e => e.className === 'ds-forge').at(-1);
  assert.equal(restoredBox.children.find(e => e.tag === 'pre').textContent, 'draft 4');
  restoredBox.children[0].children[1].onclick();
  const wrongMode = { id: 9, properties: JSON.parse(JSON.stringify(node.properties)), __dasiwaH3Forge: { ...node.__dasiwaH3Forge, mode: () => 'REF2VA' } };
  await context.window.DaSiWaH3Forge.open(wrongMode);
  assert.equal(elements('button', 'Apply to node').at(-1).disabled, true);
  all.filter(e => e.className === 'ds-forge').at(-1).children[0].children[1].onclick();
  // The Director toolbar Clear goes through the same Forge cleanup hook.
  const director = fs.readFileSync('js/minimax_h3_director.js', 'utf8');
  const clearCode = director.slice(director.indexOf('  const clearAll = () => {'), director.indexOf('  // --- Reference-pack save/load ---'));
  const clearState = { items: [{ type: 'image' }], prompt_blocks: ['old'], refmods: [{ enabled: true }] };
  const clearContext = { node, window: { DaSiWaH3Forge: context.window.DaSiWaH3Forge },
    promptWidget: { value: 'old prompt', callback() {} }, resetBuilderState() {},
    mutate(fn) { fn(clearState); }, updateRefModActiveBadge() {}, setStatus() {} };
  vm.runInNewContext(`${clearCode}\nclearAll();`, clearContext);
  assert.equal(node.properties.dasiwaH3ForgeHistory, undefined);
  assert.equal(clearContext.promptWidget.value, '');
  assert.equal(clearState.items.length, 0);
  assert.equal(clearState.refmods[0].enabled, false);
  // The Forge-only Clear history button also works without clearing the Director.
  await context.window.DaSiWaH3Forge.open(restored);
  const historyOnly = all.filter(e => e.className === 'history').at(-1);
  historyOnly.children[0].children[1].onclick();
  assert.equal(restored.properties.dasiwaH3ForgeHistory, undefined);
  assert.equal(context.forgeHistory({ properties: {} }).length, 0);
  console.log('H3 Forge history retention and restore passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
