const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function helpers() {
  const file = path.join(__dirname, '..', 'js', 'minimax_h3_forge_state.js');
  const source = fs.existsSync(file) ? fs.readFileSync(file, 'utf8').replace(/^export /gm, '') : '';
  return vm.runInNewContext(source + '\n({ forgeReferences: typeof forgeReferences === "function" ? forgeReferences : null, referenceInstructions: typeof referenceInstructions === "function" ? referenceInstructions : null, refPromptFields: typeof refPromptFields === "function" ? refPromptFields : null, refTemplate: typeof refTemplate === "function" ? refTemplate : null, inheritedDefinitions: typeof inheritedDefinitions === "function" ? inheritedDefinitions : null, referenceSnapshot: typeof referenceSnapshot === "function" ? referenceSnapshot : null, referenceTags: typeof referenceTags === "function" ? referenceTags : null })');
}

test('pose intent survives a serialized timeline and is not a new subject group', () => {
  const { forgeReferences } = helpers();
  assert.equal(typeof forgeReferences, 'function');
  const items = JSON.parse(JSON.stringify([{ id: 'pose', type: 'image', value: 'pose.png', slot: 0, forge_role: 'pose', forge_instructions: 'Arms only', forge_keep: 'stance', forge_drop: 'face', forge_subject_group: 'A' }]));
  const ref = forgeReferences(items, 'REF2VA')[0];
  assert.equal(ref.role, 'pose');
  assert.equal(ref.instructions, 'Arms only');
  assert.equal(ref.keep, 'stance');
  assert.equal(ref.drop, 'face');
  assert.equal(ref.subject_group, '');
});

test('reference numbering follows backend type then slot even for audio-only videos', () => {
  const refs = helpers().forgeReferences([
    { id: 'audio', type: 'audio', value: 'a.wav', slot: 0 },
    { id: 'v2', type: 'video', value: 'v2.mp4', slot: 2, media_mode: 'audio' },
    { id: 'v1', type: 'video', value: 'v1.mp4', slot: 1, media_mode: 'video', audio: 'attached.wav' },
    { id: 'off', type: 'image', value: 'off.png', enabled: false },
    { id: 'picture', type: 'image', value: 'p.png', slot: 4 },
  ], 'REF2VA');
  assert.equal(refs.map(r => r.item.id).join(','), 'picture,v1,v2,audio');
  assert.equal(refs[1].stream, 'both');
  assert.equal(refs[2].stream, 'audio');
});

test('template keeps free next-action text and has all six executable headings', () => {
  const { refTemplate, refPromptFields } = helpers();
  assert.equal(typeof refTemplate, 'function');
  const text = refTemplate('She walks forward.', '<Subject 1> A woman.');
  const fields = refPromptFields(text);
  assert.equal(fields.subject_definitions, '<Subject 1> A woman.');
  assert.equal(fields.detailed_description, 'She walks forward.');
  assert.equal(fields.music, '');
  assert.equal(Object.keys(fields).length, 6);
  assert.equal(refTemplate(text), text);
  assert.equal(refPromptFields(refTemplate(text, 'Reviewed identity')).subject_definitions, 'Reviewed identity');
  assert.equal(refPromptFields(refTemplate('Walk')).music, '');
  assert.ok(refPromptFields(text.replace('overall_soundscape:', 'soundscape:').replace('non_diegetic_music:', 'music:')));
});

test('mixed video streams assign paired audio first, like native conditioning', () => {
  const { referenceSnapshot, referenceTags } = helpers();
  const refs = [{ kind: 'video', stream: 'audio', id: 'voice' }, { kind: 'video', stream: 'both', id: 'av' }, { kind: 'audio', id: 'music' }];
  const snapshot = referenceSnapshot(refs);
  assert.ok(snapshot['<Audio 1>'].startsWith('av:'));
  assert.ok(snapshot['<Audio 2>'].startsWith('voice:'));
  assert.ok(snapshot['<Audio 3>'].startsWith('music:'));
  assert.equal(JSON.stringify(referenceTags(refs)), JSON.stringify([['<Audio 2>'], ['<Video 1>', '<Audio 1>'], ['<Audio 3>']]));
});

test('inherited identity citations remap automatically after image reorder', () => {
  const { inheritedDefinitions } = helpers();
  assert.equal(typeof inheritedDefinitions, 'function');
  const previous = { '<Picture 1>': 'alice', '<Picture 2>': 'bob' };
  const current = { '<Picture 1>': 'bob', '<Picture 2>': 'alice' };
  const result = inheritedDefinitions('<Subject 1> Alice from <Picture 1>.\n<Subject 2> Bob from <Picture 2>.', previous, current);
  assert.equal(result.text, '<Subject 1> Alice from <Picture 2>.\n<Subject 2> Bob from <Picture 1>.');
  assert.equal(result.warning, '');
});

test('missing or unknown reference mapping removes citations, not approved identities', () => {
  const result = helpers().inheritedDefinitions('<Subject 7> Alice: red hair, <Picture 1>.', {}, {});
  assert.ok(result.text.includes('<Subject 7> Alice: red hair'));
  assert.ok(!result.text.includes('<Picture 1>'));
  assert.ok(result.warning);
});

function directorHook({ items = [], continuing = false, useReferences = false, prompt = '' } = {}) {
  const source = fs.readFileSync(path.join(__dirname, '..', 'js', 'minimax_h3_director.js'), 'utf8');
  const start = source.indexOf('  const forgeItems =');
  const end = source.indexOf('  if (modeWidget)', start);
  assert.ok(start >= 0 && end > start);
  const state = { items, refmods: [] };
  const builderState = { simple_prompt: prompt, ref: {} };
  const c = { continuation_prompt: '', use_references: useReferences, idea: '' };
  const node = { properties: {} };
  let emitted = 0;
  const contextKey = () => JSON.stringify([state, builderState, c]);
  const ctx = { ...helpers(), state, builderState, node,
    activeItems: () => state.items.filter(i => i.enabled !== false), isLockedSlot: () => false,
    laneForItem: i => i.type, mode: () => 'REF2VA', promptStyle: () => 'simple',
    setStatus: () => {}, continuityContext: () => continuing ? c : null,
    continuityState: () => c, isContinuing: () => continuing,
    activePrompt: () => continuing ? c.continuation_prompt : builderState.simple_prompt,
    setActivePrompt: value => { if (continuing) c.continuation_prompt = value; else builderState.simple_prompt = value; },
    forgeContextKey: contextKey, refModLibrary: { entries: [], loaded: true },
    emit: () => emitted++, render: () => {}, requestAnimationFrame: () => {},
  };
  vm.runInNewContext(source.slice(start, end), ctx);
  return { hook: node.__dasiwaH3Forge, continuityHook: node.__dasiwaH3Continuity, state, builderState, c, node, refModLibrary: ctx.refModLibrary, emitted: () => emitted };
}

test('Director reference updates persist canonical state and portable packs', () => {
  const { hook, state, emitted } = directorHook({ items: [{ id: 'a', type: 'image', value: 'a.png', slot: 0 }] });
  const copy = hook.items()[0];
  assert.notEqual(copy, state.items[0]);
  hook.updateReference(copy.id, { forge_role: 'pose', forge_instructions: 'Hands only', forge_drop: 'clothes' });
  assert.equal(state.items[0].forge_role, 'pose');
  assert.equal(hook.references()[0].instructions, 'Hands only');
  assert.equal(emitted(), 1);
  const source = fs.readFileSync(path.join(__dirname, '..', 'js', 'minimax_h3_director.js'), 'utf8');
  const portable = vm.runInNewContext(source.match(/  const PORTABLE_ITEM_KEYS = .*;\n  const toPortableItem = .*;/)[0] + '\ntoPortableItem(item)', { item: state.items[0] });
  assert.equal(portable.forge_instructions, 'Hands only');
  assert.equal(portable.forge_drop, 'clothes');
});

test('cleared picture extras stay absent through persistence and a legacy draft round trip', () => {
  const item = { id: 'a', type: 'image', value: 'a.png', slot: 0, forge_label: 'character-1', forge_kinds: 'character,place', forge_who: '1' };
  const { hook, state } = directorHook({ items: [item] });
  const ref = hook.references()[0];
  const source = fs.readFileSync(path.join(__dirname, '..', 'js', 'minimax_h3_forge.js'), 'utf8');
  const start = source.indexOf('  const persistReference =');
  const end = source.indexOf('  let includeReferences', start);
  const ctx = vm.createContext({ hook, ref, patch: { forge_label: 'character-1', forge_kinds: '', forge_who: '', forge_who_axis: '' }, openedKey: hook.contextKey() });
  vm.runInContext(source.slice(start, end) + '\npersistReference(ref, patch);', ctx);
  assert.ok(!('forge_kinds' in state.items[0]));
  assert.ok(!('forge_who' in ref.item));
  const restored = directorHook({ items: JSON.parse(JSON.stringify(state.items)) });
  assert.equal(hook.contextKey(), restored.hook.contextKey());
  assert.equal(JSON.stringify(hook.references().map(({item, ...r}) => r)), JSON.stringify(restored.hook.references().map(({item, ...r}) => r)));
  const entry = { mode: 'REF2VA', continuity: false, contextKey: hook.contextKey(), simple_prompt: 'Legacy draft', fields: { ref: {} } };
  assert.equal(restored.hook.apply(entry), true);
});

test('continuity reference switch gates Forge media without deleting timeline' , () => {
  const { hook, state, c } = directorHook({ items: [{ id: 'a', type: 'image', value: 'a.png', slot: 0 }], continuing: true });
  assert.equal(hook.references().length, 0);
  assert.equal(state.items.length, 1);
  c.use_references = true;
  assert.equal(hook.references().length, 1);
});

test('template insertion and applying structured draft modify continuation only', () => {
  const { hook, continuityHook, builderState, c } = directorHook({ continuing: true, prompt: 'Original base story' });
  c.continuation_prompt = 'She looks left.';
  builderState.simple_prompt = helpers().refTemplate('Original base story', '<Subject 1> Alice');
  continuityHook.insertTemplate();
  assert.equal(helpers().refPromptFields(c.continuation_prompt).detailed_description, 'She looks left.');
  assert.equal(helpers().refPromptFields(builderState.simple_prompt).detailed_description, 'Original base story');
  const result = { mode: 'REF2VA', continuity: true, contextKey: hook.contextKey(), simple_prompt: helpers().refTemplate('She waves.', '<Subject 1> Alice'), reference_snapshot: {} };
  assert.equal(hook.apply(result), true);
  assert.equal(helpers().refPromptFields(builderState.simple_prompt).detailed_description, 'Original base story');
  assert.equal(helpers().refPromptFields(c.continuation_prompt).subject_definitions, '<Subject 1> Alice');
  assert.equal(hook.apply(result), false);
});

test('Director Continuity template needs neither Forge nor an LLM and never changes base text', () => {
  const { continuityHook, node, builderState, c } = directorHook({ continuing: true, prompt: helpers().refTemplate('Base action', '<Subject 4> Known identity.') });
  c.continuation_prompt = 'Next action only.';
  const base = builderState.simple_prompt;
  delete node.__dasiwaH3Forge;
  assert.equal(continuityHook.insertTemplate(), true);
  const fields = helpers().refPromptFields(c.continuation_prompt);
  assert.equal(fields.subject_definitions, '<Subject 4> Known identity.');
  assert.equal(fields.detailed_description, 'Next action only.');
  assert.equal(builderState.simple_prompt, base);
  const once = c.continuation_prompt;
  continuityHook.insertTemplate();
  assert.equal(c.continuation_prompt, once);
});

test('continuity template action is inactive for a new take', () => {
  const { continuityHook, builderState } = directorHook({ prompt: 'Keep unchanged' });
  assert.equal(continuityHook.insertTemplate(), false);
  assert.equal(builderState.simple_prompt, 'Keep unchanged');
});

test('template after swapping pictures remaps existing sections before snapshotting', () => {
  const { hook, continuityHook, state, builderState, c } = directorHook({ continuing: true, useReferences: true, items: [
    { id: 'alice', type: 'image', value: 'alice.png', slot: 0 },
    { id: 'bob', type: 'image', value: 'bob.png', slot: 1 },
  ], prompt: helpers().refTemplate('Alice from <Picture 1> waves to <Picture 2>.', '<Subject 1> Alice: <Picture 1>.\n<Subject 2> Bob: <Picture 2>.') });
  c.continuation_prompt = builderState.simple_prompt;
  c.forge_reference_snapshot = helpers().referenceSnapshot(hook.references());
  state.items[0].slot = 1; state.items[1].slot = 0;
  const reviewed = hook.existingDefinitions().text;
  continuityHook.insertTemplate();
  const fields = helpers().refPromptFields(c.continuation_prompt);
  assert.equal(fields.subject_definitions, reviewed);
  assert.equal(fields.detailed_description, 'Alice from <Picture 2> waves to <Picture 1>.');
  assert.equal(hook.existingDefinitions().text, reviewed);
});

test('definitions are inherited automatically but stale hidden builder fields are ignored', () => {
  const { hook, builderState } = directorHook({ continuing: true, prompt: helpers().refTemplate('Old action', '<Subject 9> Alice from <Picture 1>.') });
  const definitions = hook.existingDefinitions();
  assert.ok(definitions.text.includes('<Subject 9> Alice'));
  assert.ok(!definitions.text.includes('Old action'));
  assert.ok(!definitions.text.includes('<Picture 1>'));
  builderState.simple_prompt = 'Replacement unstructured story';
  builderState.ref.subject_definitions = 'Obsolete character';
  assert.equal(hook.existingDefinitions().text, '');
});

test('new takes never inherit subjects from a previous applied Forge prompt', () => {
  const { hook, builderState } = directorHook({ prompt: helpers().refTemplate('Old action', '<Subject 9> Alice from <Picture 1>.') });
  builderState.ref.subject_definitions = 'Obsolete hidden character';
  assert.equal(hook.existingDefinitions().text, '');
  assert.equal(hook.existingDefinitions().warning, '');
});

function forgeDefinitionContext(continuity) {
  const source = fs.readFileSync(path.join(__dirname, '..', 'js', 'minimax_h3_forge.js'), 'utf8');
  const start = source.indexOf('  const inherited =');
  const end = source.indexOf('  const structured =', start);
  return { source, context: vm.createContext({ mode: 'REF2VA', continuity,
    hook: { existingDefinitions: () => ({ text: '<Subject 1> Current identity', warning: '' }) },
  }), initialization: source.slice(start, end) };
}

test('Forge only reads inherited identities in Continuity, even with a legacy hook', () => {
  for (const continuity of [null, { use_references: true }]) {
    const { context, initialization } = forgeDefinitionContext(continuity);
    vm.runInContext(initialization, context);
    assert.equal(vm.runInContext('definitions.value', context), continuity ? '<Subject 1> Current identity' : '');
  }
});

test('selecting saved drafts never replaces current identities with historical ones', () => {
  for (const continuity of [null, { use_references: true }]) {
    const { source, context, initialization } = forgeDefinitionContext(continuity);
    Object.assign(context, { closed: false, brief: {}, structured: {}, seePictures: {}, output: {}, applyBtn: {},
      loadingModels: false, running: null, compatible: () => true, renderHistory: () => {},
      entry: { brief: 'New idea', simple_prompt: 'Saved draft', existing_definitions: '<Subject 9> Historical identity' },
    });
    vm.runInContext(initialization + '\nlet result = null;\n' + source.slice(source.indexOf('  const showResult ='), source.indexOf('  const renderHistory =')) + '\nshowResult(entry);', context);
    assert.equal(vm.runInContext('definitions.value', context), continuity ? '<Subject 1> Current identity' : '');
    assert.equal(context.output.textContent, 'Saved draft');
  }
});

test('a replacement file in the same timeline slot never inherits the old media citation', () => {
  const { referenceSnapshot, inheritedDefinitions, forgeReferences } = helpers();
  const before = referenceSnapshot(forgeReferences([{ id: 'v', type: 'video', value: 'old.mp4', slot: 0 }], 'REF2VA'));
  const after = referenceSnapshot(forgeReferences([{ id: 'v', type: 'video', value: 'new.mp4', slot: 0 }], 'REF2VA'));
  const result = inheritedDefinitions('<Subject 1> Alice from <Video 1>.', before, after);
  assert.ok(!result.text.includes('<Video 1>'));
});

test('Forge canvas uses Director output widgets, not reference image dimensions', () => {
  const { hook, node } = directorHook({ items: [{ id: 'ref', type: 'image', value: 'wide.png', width: 1920, height: 1080 }] });
  node.widgets = [{ name: 'width', value: 768 }, { name: 'height', value: 1360 }];
  assert.equal(JSON.stringify(hook.outputCanvas()), JSON.stringify({ width: 768, height: 1360 }));
  node.widgets[0].value = 1024;
  assert.equal(hook.outputCanvas().width, 1024);
  node.widgets[0].value = 0;
  assert.equal(hook.outputCanvas(), null);
});

test('either linked external canvas override makes Forge dimensions unknown', () => {
  const { hook, node } = directorHook();
  node.widgets = [{ name: 'width', value: 1280 }, { name: 'height', value: 720 }];
  for (const name of ['external_width_overwrite', 'external_height_overwrite']) {
    node.inputs = [{ name, link: 0 }];
    assert.equal(hook.outputCanvas(), null);
    node.inputs[0].link = null;
    assert.equal(hook.outputCanvas().width, 1280);
  }
});

test('Forge request forwards known canvas and explicitly sends null when unknown', () => {
  const source = fs.readFileSync(path.join(__dirname, '..', 'js', 'minimax_h3_forge.js'), 'utf8');
  const expression = source.match(/output_canvas: ([^\n]+),/)[1];
  for (const canvas of [{ width: 1280, height: 720 }, null]) {
    const payload = vm.runInNewContext(`({ output_canvas: ${expression} })`, { hook: { outputCanvas: () => canvas } });
    assert.equal(JSON.stringify(payload.output_canvas), JSON.stringify(canvas));
  }
});

test('missing saved references are skipped just as runtime does', () => {
  const { hook, state } = directorHook();
  state.refmods = [{ name: 'missing.safetensors', slot: 1, media_type: 'image', enabled: true }];
  assert.equal(hook.references().length, 0);
});

test('saved multi-image members retain distinct identities and native kind numbering', () => {
  const { hook, state, refModLibrary } = directorHook({ items: [{ id: 'a', type: 'image', value: 'a.png', slot: 0 }] });
  state.refmods = [{ name: 'bundle.safetensors', slot: 1, enabled: true, description: 'Alice' }];
  refModLibrary.entries = [{ name: 'bundle.safetensors', kinds: ['image', 'image', 'audio'] }];
  const refs = hook.references();
  const snapshot = helpers().referenceSnapshot(refs);
  assert.equal(refs.length, 4);
  assert.notEqual(snapshot['<Picture 2>'], snapshot['<Picture 3>']);
});
