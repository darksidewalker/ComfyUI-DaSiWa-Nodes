import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const source = (await readFile(new URL("../js/director_refmods.js", import.meta.url), "utf8"))
  .replace('import { api } from "../../scripts/api.js";', 'const api = {};');
const { refModPreview } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const library = [{name:"bilge",kind:"video"},{name:"ahmet",kind:"video"}];
const state = {items:[{type:"video",value:"scene.mp4"},{type:"video",value:"audio.mp4",media_mode:"audio"}],
  refmods:[{slot:1,name:"bilge",description:"Bilge"},{slot:3,name:"ahmet",description:"Ahmet"}]};
assert.equal(refModPreview("<RefMod 1> greets <RefMod 3> in <Video 1>.",state,library),
  "<Video 2> greets <Video 3> in <Video 1>.\n\nReference descriptions:\n<Video 2>: Bilge\n<Video 3>: Ahmet");
state.refmods[0].enabled=false;
assert.match(refModPreview("<RefMod 3>",state,library),/^<Video 2>/);
assert.throws(()=>refModPreview("<RefMod 1>",state,library),/no active reference/);
assert.throws(()=>refModPreview("<RefMod 3>",state,[]),/Refresh/);
assert.equal(refModPreview("plain prompt",{items:[]},[]),"plain prompt");
console.log("RefMod preview mapping, stable slots, disabled slots and plain prompts: PASS");
