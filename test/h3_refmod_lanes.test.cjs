const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('js/minimax_h3_director.js', 'utf8');
const start = source.indexOf('  const activeItems = () =>');
const end = source.indexOf('  const ensureLayout =', start);
assert.ok(start >= 0 && end > start);
const context = {
  state: {
    items: [
      { id: 'photo', type: 'image', slot: 0 },
      { id: 'clip', type: 'video', slot: 0, media_mode: 'video_audio', audioSlot: 2 },
      { id: 'sound', type: 'audio', slot: 0 },
    ],
    refmods: [{ slot: 1, name: 'bundle', enabled: true, strength: 1 },
      { slot: 2, name: 'voice', enabled: true, strength: 0.5 }],
  },
  refModLibrary: { entries: [{ name: 'bundle', kinds: ['image', 'video', 'audio'] },
    { name: 'voice', kind: 'audio' }] },
  mode: () => 'REF2VA',
  MAX: { image: 9, video: 3, audio: 3 },
};
const helpers = source.slice(source.indexOf('  const laneForItem ='), source.indexOf('  const availableSlots ='));
vm.runInNewContext(source.slice(start, end) + '\n' + helpers + '\nthis.entries = refmodTimelineItems();', context);
assert.deepEqual(Array.from(context.entries, item => [item.type, item.slot]),
  [['image', 9], ['video', 3], ['audio', 3], ['audio', 4]]);
assert.deepEqual(Array.from(context.state.items, item => item.slot), [0, 0, 0]);
assert.equal(context.entries.every(item => item._isRefMod), true);
assert.equal(context.entries[0].refmodSlot, 1);
assert.equal(context.entries[3].refmodSlot, 2);
console.log('Mixed uploaded media and RefMod bundles occupy separate display slots per lane');
