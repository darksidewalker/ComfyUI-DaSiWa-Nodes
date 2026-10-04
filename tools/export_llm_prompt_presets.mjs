#!/usr/bin/env node
// Offline developer export; PromptForge is not a runtime dependency.
// Usage: node tools/export_llm_prompt_presets.mjs PROMPTFORGE_ROOT OUT_JSON
import { writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';

const [rootArg, outArg] = process.argv.slice(2);
if (!rootArg || !outArg) {
  throw new Error('Usage: node tools/export_llm_prompt_presets.mjs PROMPTFORGE_ROOT OUT_JSON');
}
const root = resolve(rootArg);
const load = rel => import(pathToFileURL(join(root, rel)).href);
const { loadConfig } = await load('server/config.mjs');
const { loadRegistry } = await load('server/registry.mjs');
const { createPromptStore } = await load('server/prompts.mjs');
const { config, error } = loadConfig(root, 'server');
if (error || !config) throw new Error(error || 'Missing PromptForge server config');
const registry = loadRegistry(root);
const store = createPromptStore(root, config, registry);
store.reload();
const selections = [
  ['promptforge_wan22', 'wan', 'Positive prompt'],
  ['promptforge_ltx', 'ltx', 'Enhanced paragraph'],
  ['promptforge_krea2', 'krea2', 'Enhanced prompt'],
  ['promptforge_anima', 'anima', 'Positive prompt'],
  ['promptforge_illustrious', 'illustrious', 'Positive prompt'],
];
const presets = {};
for (const [id, key, output] of selections) {
  const model = registry.get(key);
  if (!model || !model.segments.includes(output)) {
    throw new Error(`Missing output ${key}:${output}`);
  }
  const system = [store.get(key), store.segmentTail(model, [output])]
    .filter(Boolean).join('\n\n---\n\n');
  if (!system.trim()) throw new Error(`Empty system ${key}`);
  presets[id] = { model: key, output, segments: [output], system };
}
const revision = execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], {
  encoding: 'utf8',
}).trim();
const bundle = { schema_version: 1, source: 'PromptForge', source_revision: revision, presets };
writeFileSync(resolve(outArg), JSON.stringify(bundle, null, 2) + '\n', 'utf8');
console.log(`Exported ${Object.keys(presets).length} presets`);
