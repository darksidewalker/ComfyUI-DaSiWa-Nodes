const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Minimal DOM host; all dialog behavior runs from the real extension.
class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.style = {};
    this.listeners = {}; this.attributes = {}; this.hidden = false;
    this.disabled = false; this.checked = false; this.value = ''; this._text = '';
    this.classList = { toggle() {} };
  }
  set textContent(value) { this._text = value; }
  get textContent() { return this._text + this.children.map(c => typeof c === 'string' ? c : c.textContent).join(''); }
  append(...children) {
    for (const child of children) { this.children.push(child); if (typeof child !== 'string') child.parentElement = this; }
  }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  insertBefore(child, before) { const i = this.children.indexOf(before); this.children.splice(i, 0, child); child.parentElement = this; }
  remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(c => c !== this); }
  focus() {}
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
  async dispatch(type) {
    const event = { target: this };
    await this['on' + type]?.(event);
    for (const callback of this.listeners[type] || []) await callback(event);
  }
  querySelectorAll(selector) {
    const descendants = this.children.flatMap(c => typeof c === 'string' ? [] : [c, ...c.querySelectorAll('*')]);
    if (selector === '*') return descendants;
    return descendants.filter(c => selector.split(',').some(s => {
      const parts = s.trim().split(' '); const tag = parts.pop();
      if (c.tagName !== tag.toUpperCase()) return false;
      if (!parts.length) return true;
      for (let parent = c.parentElement; parent; parent = parent.parentElement) {
        if (parent.className?.split(' ').includes(parts[0].slice(1))) return true;
      }
      return false;
    }));
  }
  get options() { return this.querySelectorAll('option'); }
  get selectedOptions() { return this.options.filter(o => o.value === this.value); }
}
const root = path.join(__dirname, '..');
const image = (role = 'character-1') => ({ kind: 'image', path: 'a.png', easy_role: role, role: 'subject', instructions: '', item: { id: 'a' } });
async function dialog({ refs = [image()], mode = 'REF2VA', continuity = null, history = [], prefs = {} } = {}) {
  const document = { head: new Element('head'), body: new Element('body'), createElement: t => new Element(t), getElementById: () => null, addEventListener() {}, removeEventListener() {} };
  const requests = [], applied = [];
  const node = { id: 1, properties: { dasiwaH3ForgeHistory: history }, graph: { setDirtyCanvas() {} }, __dasiwaH3Forge: {
    references: () => refs, mode: () => mode, continuity: () => continuity, contextKey: () => 'context', duration: () => 5,
    updateReference: () => true, apply: entry => { applied.push(entry); return true; }, setStatus() {},
  } };
  let stored = JSON.stringify(prefs);
  const context = vm.createContext({ document, window: {}, app: { registerExtension() {} },
    localStorage: { getItem: () => stored, setItem: (_, value) => { stored = value; } },
    setInterval: () => 1, clearInterval() {},
    api: { apiURL: p => p, fetchApi: async (url, options) => {
      if (url.endsWith('/models')) return { ok: true, json: async () => ({ models: [{ id: 'local:test', label: 'Test' }], creativity: ['balanced'], default_creativity: 'balanced', detail_levels: {}, default_detail: 5, shot_counts: ['Auto'] }) };
      const request = JSON.parse(options.body); requests.push(request);
      return { ok: true, json: async () => ({ mode, continuity: !!continuity, source_id: continuity?.clip_id, simple_prompt: 'Generated draft', fields: {}, model: 'local:test', easy: request.easy, vision: true, saw_images: request.see_pictures ? 1 : 0, unloaded: true, stats: { seconds: 1 } }) };
    } },
  });
  const state = fs.readFileSync(path.join(root, 'js/minimax_h3_forge_state.js'), 'utf8').replace(/^export /gm, '');
  const extension = fs.readFileSync(path.join(root, 'js/minimax_h3_forge.js'), 'utf8').replace(/^import .*;\n/gm, '');
  vm.runInContext(state + '\n' + extension, context);
  await context.window.DaSiWaH3Forge.open(node);
  const all = document.body.querySelectorAll('*');
  const button = name => all.find(c => c.tagName === 'BUTTON' && c.textContent === name);
  const choice = all.find(c => c.tagName === 'LABEL' && c.textContent.includes('Let the model see the pictures'));
  const visible = element => !!element && !element.hidden && (!element.parentElement || visible(element.parentElement));
  return { node, refs, requests, applied, all, button, choice, checkbox: choice?.children[0], visible,
    brief: all.find(c => c.tagName === 'TEXTAREA'), historyButton: () => all.find(c => c.title === 'Show this prompt; Apply to node to use it'),
    close: () => context.window.DaSiWaH3Forge.close(node) };
}

test('eligibility accepts exactly the backend EASY_ROLES whitelist', async () => {
  const roles = [...Array.from({ length: 32 }, (_, i) => `character-${i + 1}`), 'place', 'style', 'first-frame', 'last-frame', 'pose', 'custom', 'group-12', 'group-21', 'group-13', 'group-31', 'group-23', 'group-32', 'group-123'];
  for (const role of [...roles, 'character-0', 'character-33', 'group-321', 'unknown', '']) {
    const d = await dialog({ refs: [image(role)] });
    assert.equal(d.visible(d.choice), roles.includes(role), role);
    d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
    assert.equal(d.requests[0].easy, roles.includes(role), role);
    d.close();
  }
});

test('no references, base modes and continuity cannot opt into easy writer vision', async () => {
  for (const options of [{ refs: [] }, { mode: 'I2VA' }, { mode: 'T2VA' }, { continuity: { clip_id: 'c', use_references: true } }]) {
    const d = await dialog({ ...options, prefs: { see_pictures: true } });
    assert.equal(d.visible(d.choice), false);
    d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
    assert.equal(d.requests[0].easy, false);
    assert.equal(d.requests[0].see_pictures, false);
    d.close();
  }
});

test('history restores per-draft vision options and missing see_pictures defaults blind', async () => {
  for (const vision of [undefined, false, true]) {
    const options = { model: 'local:test', detail: 5, creativity: 'balanced', shots: 'Auto' };
    if (vision !== undefined) options.see_pictures = vision;
    const history = [{ mode: 'REF2VA', simple_prompt: 'Saved draft', fields: {}, brief: 'An idea', draftOptions: options }];
    const d = await dialog({ history, prefs: { see_pictures: !vision } });
    await d.historyButton().dispatch('click');
    assert.equal(d.checkbox.checked, !!vision);
    assert.equal(d.button('Apply to node').disabled, false);
    await d.button('Apply to node').dispatch('click');
    assert.equal(d.applied.length, 1);
    d.close();
  }
});

test('a vision history entry is incompatible when the current references are mixed', async () => {
  const history = [{ mode: 'REF2VA', simple_prompt: 'Vision draft', fields: {}, brief: 'An idea', draftOptions: { see_pictures: true, shots: 'Auto' } }];
  const d = await dialog({ refs: [image(), { kind: 'audio', instructions: '' }], history });
  await d.historyButton().dispatch('click');
  assert.equal(d.button('Apply to node').disabled, true);
  await d.button('Apply to node').dispatch('click');
  assert.equal(d.applied.length, 0);
  d.close();
});

test('labelling an initially unlabelled image enables vision choice without reopening', async () => {
  const ref = image();
  delete ref.easy_role;
  const reopened = await dialog({ refs: [ref] });
  assert.equal(reopened.visible(reopened.choice), false);
  const selector = reopened.all.find(c => c.attributes['aria-label']?.endsWith(' label'));
  assert.ok(selector, 'unlabelled image still has a label editor');
  selector.value = 'place'; await selector.dispatch('change');
  assert.equal(reopened.visible(reopened.choice), true);
  reopened.checkbox.checked = true;
  reopened.brief.value = 'A place'; await reopened.button('Generate').dispatch('click');
  assert.equal(reopened.requests[0].easy, true);
  assert.equal(reopened.requests[0].see_pictures, true);
  reopened.close();
});

test('apply rejects a vision mismatch even without a dispatched change event', async () => {
  const d = await dialog();
  d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
  const saved = d.node.properties.dasiwaH3ForgeHistory[0];
  assert.equal(JSON.parse(saved.forgeInputKey).length, 4, 'legacy input key shape is unchanged');
  d.checkbox.checked = true;
  await d.button('Apply to node').dispatch('click');
  assert.equal(d.applied.length, 0);
  d.close();
});

test('legacy history without draftOptions restores blind despite a vision preference', async () => {
  const history = [{ mode: 'REF2VA', simple_prompt: 'Old draft', fields: {}, brief: 'An idea' }];
  const d = await dialog({ history, prefs: { see_pictures: true } });
  await d.historyButton().dispatch('click');
  assert.equal(d.checkbox.checked, false);
  assert.equal(d.button('Apply to node').disabled, false);
  d.close();
});

test('changing the vision checkbox invalidates a generated draft and blocks apply', async () => {
  const d = await dialog();
  d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
  assert.equal(d.button('Apply to node').disabled, false);
  d.checkbox.checked = true; await d.checkbox.dispatch('change');
  assert.equal(d.button('Apply to node').disabled, true);
  assert.equal(d.all.find(c => c.tagName === 'PRE').hidden, true);
  await d.button('Apply to node').dispatch('click');
  assert.equal(d.applied.length, 0);
  d.close();
});

test('mixed references hide the opt-out while keeping label editors and using normal vision', async () => {
  const d = await dialog({ refs: [image(), { kind: 'video', path: 'v.mp4', stream: 'video', instructions: '' }], prefs: { see_pictures: true } });
  assert.equal(d.visible(d.choice), false);
  assert.ok(d.all.some(c => c.attributes['aria-label']?.endsWith(' label')));
  assert.ok(d.all.some(c => d.visible(c) && c.textContent.includes('Pictures are automatically sent to a model capable')));
  d.brief.value = 'An idea'; await d.button('Generate').dispatch('click');
  assert.equal(d.requests[0].easy, false); assert.equal(d.requests[0].see_pictures, false);
  d.close();
});
