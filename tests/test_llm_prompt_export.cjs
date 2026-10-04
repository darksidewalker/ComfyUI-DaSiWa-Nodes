'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const repo = path.resolve(__dirname, '..');
const artifact = path.join(repo, 'data/llm_prompt_presets.json');
const selections = [
  ['promptforge_wan22', 'wan', 'Positive prompt'],
  ['promptforge_ltx', 'ltx', 'Enhanced paragraph'],
  ['promptforge_krea2', 'krea2', 'Enhanced prompt'],
  ['promptforge_anima', 'anima', 'Positive prompt'],
  ['promptforge_illustrious', 'illustrious', 'Positive prompt'],
];

test('offline presets have provenance and only non-H3 primary outputs', () => {
  const bundle = JSON.parse(fs.readFileSync(artifact, 'utf8'));
  assert.equal(bundle.schema_version, 1);
  assert.equal(bundle.source, 'PromptForge');
  assert.match(bundle.source_revision, /^[0-9a-f]{40}$/);
  assert.deepEqual(Object.keys(bundle.presets), selections.map(([id]) => id));
  assert.equal(bundle.presets.promptforge_h3, undefined);
  assert.equal(bundle.exported_at, undefined);
  for (const [id, model, output] of selections) {
    const preset = bundle.presets[id];
    assert.equal(preset.model, model);
    assert.equal(preset.output, output);
    assert.deepEqual(preset.segments, [output]);
    assert.ok(preset.system.includes(`===SEGMENT: ${output}`), id);
    assert.ok(preset.system.trim(), id);
  }
});

const referenceRoot = process.env.PROMPTFORGE_SOURCE_ROOT;
test('export is reproducible and equals the actual PromptForge store', { skip: !referenceRoot }, async () => {
  const { pathToFileURL } = require('node:url');
  const { execFileSync } = require('node:child_process');
  const scratch = process.env.TMPDIR;
  assert.ok(scratch, 'Set TMPDIR to the managed scratch workspace');
  const directory = fs.mkdtempSync(path.join(scratch, 'llm-export-'));
  try {
    const first = path.join(directory, 'first.json');
    const second = path.join(directory, 'second.json');
    const hookArgs = process.execArgv.filter((_, i, args) => args[i - 1] === '--import' || args[i] === '--import');
    for (const output of [first, second]) {
      execFileSync(process.execPath, [...hookArgs, path.join(repo, 'tools/export_llm_prompt_presets.mjs'), referenceRoot, output]);
    }
    assert.equal(fs.readFileSync(first, 'utf8'), fs.readFileSync(second, 'utf8'));
    assert.equal(fs.readFileSync(first, 'utf8'), fs.readFileSync(artifact, 'utf8'));
    const load = rel => import(pathToFileURL(path.join(referenceRoot, rel)).href);
    const { loadConfig } = await load('server/config.mjs');
    const { loadRegistry } = await load('server/registry.mjs');
    const { createPromptStore } = await load('server/prompts.mjs');
    const { config } = loadConfig(referenceRoot, 'server');
    const registry = loadRegistry(referenceRoot);
    const store = createPromptStore(referenceRoot, config, registry);
    store.reload();
    const bundle = JSON.parse(fs.readFileSync(artifact, 'utf8'));
    for (const [id, key, output] of selections) {
      const actual = [store.get(key), store.segmentTail(registry.get(key), [output])].filter(Boolean).join('\n\n---\n\n');
      assert.equal(bundle.presets[id].system, actual, id);
    }
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
