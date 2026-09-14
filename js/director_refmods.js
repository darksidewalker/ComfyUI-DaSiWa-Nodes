import { api } from "../../scripts/api.js";

export function refModPreview(text, state, entries) {
  const counts = { image: 0, video: 0 };
  for (const item of state.items || []) {
    if (item.enabled === false || !item.value) continue;
    if (item.type === "image") counts.image++;
    if (item.type === "video" && item.media_mode !== "audio") counts.video++;
  }
  const tags = new Map(), descriptions = [];
  const rows = (state.refmods || []).filter(r => r.name && r.enabled !== false && Number(r.strength ?? 1) > 0).sort((a, b) => a.slot - b.slot);
  for (const row of rows) {
    const entry = (entries || []).find(e => e.name === row.name);
    if (!entry) throw new Error(`Refresh the RefMod library to preview ${row.name}.`);
    const index = ++counts[entry.kind];
    const tag = `<${entry.kind === "image" ? "Picture" : "Video"} ${index}>`;
    tags.set(Number(row.slot), tag);
    if ((row.description || "").trim()) descriptions.push(`${tag}: ${row.description.trim()}`);
  }
  const replace = value => value.replace(/<\s*refmod\s*_?\s*(\d+)(?:\s*:[^>]+)?\s*>/gi, (match, slot) => {
    if (!tags.has(Number(slot))) throw new Error(`${match} has no active reference.`);
    return tags.get(Number(slot));
  });
  return replace(text) + (descriptions.length ? "\n\nReference descriptions:\n" + descriptions.map(replace).join("\n") : "");
}

export function renderRefMods({ state, mode, library, commit, insert }) {
  const panel = document.createElement("section");
  panel.className = "ds-h3-refmods";
  panel.style.cssText = "padding:12px;margin:10px 0;border:1px solid #516278;border-radius:8px;background:#17212b;color:#e0e8ef;max-height:360px;overflow:auto";
  const rows = state.refmods || [];
  const heading = document.createElement("strong");
  heading.textContent = "RefMods · People references";
  panel.append(heading);
  const help = document.createElement("p");
  help.style.cssText = "font-size:12px;margin:6px 0;color:#b5c4d1";
  help.textContent = mode !== "REF2VA" ? "Select REF2VA to use RefMods. Saved selections are preserved."
    : "Select a saved reference, edit its description, then insert its tag into the prompt. Slot numbers stay fixed.";
  panel.append(help);
  const buttons = document.createElement("div");
  buttons.style.cssText = "display:flex;gap:8px;margin-bottom:8px";
  const button = (label, action) => {
    const el = document.createElement("button"); el.type = "button"; el.textContent = label;
    el.style.cssText = "padding:5px 9px;cursor:pointer";
    el.onclick = action; return el;
  };
  const body = document.createElement("div");
  const status = document.createElement("div"); status.role = "status";
  status.style.cssText = "font-size:12px;margin:6px 0;color:#b5c4d1";
  const refresh = async () => {
    if (library.loading) return;
    library.loading = true; status.textContent = "Loading RefMod library…";
    try {
      const response = await api.fetchApi("/dasiwa/director-refmods");
      if (!response.ok) throw new Error("Library unavailable. Restart ComfyUI after installing this update.");
      const data = await response.json();
      library.entries = data.mods || []; library.error = null;
    } catch (error) { library.error = error.message; }
    finally { library.loading = false; draw(); }
  };
  const add = button("+ Add RefMod", () => {
    const used = new Set(rows.map(row => row.slot));
    const slot = Array.from({ length: 8 }, (_, i) => i + 1).find(i => !used.has(i));
    if (!slot) return;
    state.refmods = [...rows, { slot, name: "", description: "", strength: 1, enabled: true }].sort((a, b) => a.slot - b.slot);
    commit();
  });
  add.disabled = rows.length >= 8 || mode !== "REF2VA";
  buttons.append(add, button("Refresh library", refresh));
  panel.append(buttons, status, body);
  function draw() {
    body.replaceChildren();
    status.textContent = library.error || (library.entries ? `${library.entries.length} visual RefMods available` : "Library not loaded yet");
    for (const row of rows) {
      const card = document.createElement("div");
      card.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;padding:10px 0;border-top:1px solid #36495a";
      const label = document.createElement("strong"); label.textContent = `<RefMod ${row.slot}>`;
      const remove = button("Remove", () => { state.refmods = rows.filter(r => r.slot !== row.slot); commit(); });
      const select = document.createElement("select");
      select.setAttribute("aria-label", `RefMod ${row.slot} file`);
      select.style.cssText = "grid-column:1/-1;width:100%;min-width:0;padding:6px;background:#253444;color:#fff";
      select.add(new Option("Select a RefMod file…", ""));
      for (const entry of library.entries || []) select.add(new Option(entry.name, entry.name));
      if (row.name && !(library.entries || []).some(e => e.name === row.name)) select.add(new Option(`${row.name} (not in current library)`, row.name));
      select.value = row.name;
      select.onchange = () => {
        row.name = select.value;
        const entry = (library.entries || []).find(e => e.name === row.name);
        row.description = entry?.description || "";
        commit();
      };
      const desc = document.createElement("textarea"); desc.value = row.description || "";
      desc.placeholder = "Person / appearance description";
      desc.setAttribute("aria-label", `RefMod ${row.slot} description`);
      desc.rows = 2; desc.style.cssText = "grid-column:1/-1;width:100%;box-sizing:border-box;resize:vertical;background:#253444;color:#fff;padding:6px";
      desc.oninput = () => { row.description = desc.value; commit(false); };
      desc.addEventListener("keydown", event => { if (!((event.ctrlKey || event.metaKey) && event.key === "Enter")) event.stopPropagation(); });
      const controls = document.createElement("div"); controls.style.cssText = "display:flex;gap:10px;align-items:center;flex-wrap:wrap;grid-column:1/-1";
      const enabledLabel = document.createElement("label"); const enabled = document.createElement("input"); enabled.type = "checkbox"; enabled.checked = row.enabled !== false;
      enabled.onchange = () => { row.enabled = enabled.checked; commit(); }; enabledLabel.append(enabled, " Enabled");
      const strengthLabel = document.createElement("label"); strengthLabel.textContent = "Strength ";
      const strength = document.createElement("input"); strength.type = "number"; strength.min = "0"; strength.max = "1"; strength.step = "0.05"; strength.value = row.strength ?? 1;
      strength.style.width = "65px";
      strength.onchange = () => { if (strength.checkValidity() && strength.value !== "") { row.strength = Number(strength.value); commit(false); } };
      strengthLabel.append(strength);
      const tag = button("Insert tag", () => insert(`<RefMod ${row.slot}>`));
      tag.onpointerdown = event => event.preventDefault();
      tag.disabled = !row.name || row.enabled === false || Number(row.strength) === 0;
      const entry = (library.entries || []).find(e => e.name === row.name);
      controls.append(enabledLabel, strengthLabel, tag);
      if (entry) { const cost = document.createElement("span"); cost.textContent = `${entry.tokens.toLocaleString()} ref tokens`; controls.append(cost); }
      card.append(label, remove, select, desc, controls);
      if (mode !== "REF2VA") card.querySelectorAll("input,select,textarea,button").forEach(el => { if (el !== remove) el.disabled = true; });
      body.append(card);
    }
  }
  draw();
  if (!library.entries && !library.error && !library.loading && mode === "REF2VA") refresh();
  return panel;
}
