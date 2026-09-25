const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('js/minimax_h3_director.js', 'utf8');
const between = (a, b) => source.slice(source.indexOf(a), source.indexOf(b));
function harness(initial = {}) {
  const nodes = [];
  function element(tag) {
    const el = { tag, children: [], style: {}, className: '', classList: { toggle() {}, add() {} },
      append(...xs) { this.children.push(...xs); }, replaceChildren(...xs) { this.children = xs; },
      prepend(x) { this.children.unshift(x); }, addEventListener() {}, querySelector(tagName) { return this.children.find(x => x.tag === tagName); },
      setAttribute() {}, focus() {} };
    nodes.push(el); return el;
  }
  const state = { ...initial };
  const widget = { value: 'FL2VA', callback() {} };
  const ctx = { state, node: { widgets: [widget], properties: {} }, modeWidget: widget,
    mode: () => widget.value, builderState: { simple_prompt: 'new authored' },
    document: { createElement: element, createTextNode: text => ({ tag: '#text', textContent: text }) }, Option: function(text, value) { return { tag: 'option', textContent: text, value }; }, api: { fetchApi: async () => ({ ok: true, json: async () => ({ clips: [{ clip_id: 'take_1', completed_ns: 10 }, { clip_id: 'take_2', completed_ns: 9 }], latest_id: 'take_1' }) }) },
    setStatus(text, error) { ctx.message = text; ctx.error = error; }, emit() { ctx.serialized = JSON.stringify(ctx.state); }, render() { ctx.renders = (ctx.renders || 0) + 1; },
    crypto: { randomUUID: () => 'test-session' }, window: {}, console,
  };
  vm.runInNewContext(between('const DEFAULT_CONTINUITY =', 'const MAX =') + between('  function continuityState() {', '  function refPrefill(') + '\nthis.activePrompt = activePrompt;', ctx);
  return { ctx, nodes };
}

test('virtual Continue keeps backend mode, pins a source, and stores a different prompt', async () => {
  const { ctx } = harness();
  ctx.continuityState().session = 'session_1';
  await ctx.refreshContinuity();
  assert.equal(ctx.continuityState().source_id, '');
  assert.equal(ctx.enterContinue(), false);
  ctx.selectContinuitySource('take_2');
  assert.equal(ctx.enterContinue(), true);
  assert.equal(ctx.modeWidget.value, 'FL2VA');
  assert.equal(ctx.continuityState().operation, 'continue');
  ctx.setActivePrompt('next authored');
  assert.equal(ctx.activePrompt(), 'next authored');
  assert.equal(ctx.builderState.simple_prompt, 'new authored');
  ctx.selectRealMode('T2VA');
  assert.equal(ctx.continuityState().operation, 'new');
  assert.equal(ctx.activePrompt(), 'new authored');
  assert.equal(ctx.continuityState().source_id, 'take_2');
  assert.equal(ctx.modeWidget.value, 'T2VA');
  assert.equal(ctx.continuityState().overlap_frames, 22);
  assert.equal(ctx.continuityState().extension_frames, 119);
});
test('refresh never advances pinned source; explicit latest selection does', async () => {
  const { ctx } = harness({ continuity: { session: 'session_1', source_id: 'take_2', operation: 'continue' } });
  await ctx.refreshContinuity();
  assert.equal(ctx.continuityState().source_id, 'take_2');
  ctx.useLatestContinuity();
  assert.equal(ctx.continuityState().source_id, 'take_1');
  ctx.onContinuitySaved({ detail: { session: 'other', clip_id: 'take_3' } });
  assert.equal(ctx.continuityState().source_id, 'take_1');
});
test('no session or capture until explicit interaction; Image Inpaint rejects Continue', () => {
  const { ctx } = harness();
  assert.equal(ctx.state.continuity, undefined);
  ctx.modeWidget.value = 'Image Inpaint';
  ctx.selectContinuitySource('take_1');
  assert.equal(ctx.enterContinue(), false);
  assert.equal(ctx.continuityState().operation, 'new');
  assert.equal(ctx.continuityState().capture, false);
});
test('capture warns and cannot opt in when technical graph nodes are absent', () => {
  const { ctx } = harness({ continuity: { session: 'session_1', source_id: 'take_1' } });
  ctx.node.graph = { _nodes: [{ type: 'MiniMaxH3Director' }] };
  assert.equal(ctx.enterContinue(), false);
  assert.match(ctx.message, /Append.*Publish/);
  assert.equal(ctx.continuityState().capture, false);
});

test('capture requires both append and publish; one partial branch is unsafe', () => {
  const { ctx } = harness({ continuity: { session: 'session_1', source_id: 'take_1' } });
  ctx.node.graph = { _nodes: [{ type: 'DaSiWaH3ContinuityAppend' }] };
  assert.equal(ctx.enterContinue(), false);
  ctx.node.graph = { _nodes: [{ type: 'DaSiWaH3ContinuityPublish' }] };
  assert.equal(ctx.enterContinue(), false);
  ctx.node.graph = { _nodes: [
    { type: 'DaSiWaH3ContinuityAppend' }, { type: 'DaSiWaH3ContinuityPublish' },
  ] };
  assert.equal(ctx.enterContinue(), true);
});

test('draft requires explicit Apply and refuses a stale source or idea', async () => {
  const { ctx } = harness({ continuity: { operation: 'continue', capture: true, session: 's', source_id: 'take_1', idea: 'move onward' } });
  let payload;
  ctx.window.DaSiWaH3Forge = { settings: () => ({ ollama_url: 'http://localhost:11434', openai_url: '' }) };
  ctx.api.fetchApi = async (path, options) => { payload = { path, options: JSON.parse(options.body) }; return { ok: true, json: async () => ({ prompt: 'fresh draft', source_id: 'take_1' }) }; };
  await ctx.analyzeContinuity('local:model');
  assert.equal(payload.path, '/df_h3_continuity/analyze');
  assert.equal(payload.options.model, 'local:model');
  assert.equal(payload.options.clip_id, 'take_1');
  assert.equal(payload.options.settings.ollama_url, 'http://localhost:11434');
  assert.equal(ctx.activePrompt(), ctx.continuityState().continuation_prompt);
  assert.notEqual(ctx.activePrompt(), 'fresh draft');
  ctx.continuityState().idea = 'new idea';
  assert.equal(ctx.applyContinuityDraft(), false);
  ctx.continuityState().idea = 'move onward';
  assert.equal(ctx.applyContinuityDraft(), true);
  assert.equal(ctx.activePrompt(), 'fresh draft');
});

test('compact row opts into capture without creating graph nodes or an initial session', async () => {
  const { ctx } = harness();
  const parent = { children: [], append(el) { this.children.push(el); } };
  ctx.buildContinuityControls(parent);
  assert.equal(ctx.state.continuity, undefined);
  assert.equal(parent.children.length, 1);
  const capture = parent.children[0].children[0].children[0];
  assert.equal(capture.type, 'checkbox');
  capture.checked = true;
  capture.onchange();
  assert.equal(ctx.state.continuity.capture, true);
  assert.equal(ctx.state.continuity.session, 'test_session');
  assert.equal(ctx.node.widgets.length, 1);
});

test('frame controls only offer backend-valid H3 sampling windows', () => {
  assert.match(source, /\[5, 22, 39, 56, 73\]/);
  assert.match(source, /count \+= 17\) extension\.append/);
});

test('Director reuses one prompt editor and keeps external overwrite New-only', () => {
  assert.match(source, /createBuilderField\("Prompt", activePrompt\(\)/);
  assert.match(source, /if \(hasExternalPrompt\(\) && !isContinuing\(\)\)/);
  assert.match(source, /continueButton\.textContent = "Continue"/);
  assert.doesNotMatch(source, /MutationObserver|modeWidget\.value = "Continue"/);
  assert.match(source, /timeline\.style\.width = `\$\{Math\.max\(1, width - 20\)\}px`/);
});
