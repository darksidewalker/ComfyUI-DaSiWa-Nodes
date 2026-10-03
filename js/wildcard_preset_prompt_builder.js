import { app } from "../../scripts/app.js";

const NODE_TYPE = "DaSiWa_WildcardPresetPromptBuilder";
const LIBRARY_URL = "/dasiwa/wildcard-preset-prompt-builder/library";
const PICKER_HEIGHT = 490;
const TOKEN_RE = /[\p{L}\p{N}_]+(?:['’\-][\p{L}\p{N}_]+)?|[^\s\p{L}\p{N}_]/gu;
const encoder = new TextEncoder();
let installWildcardPicker;

function widget(node, name) {
    return node.widgets?.find((entry) => entry.name === name);
}

function isWildcardNode(node) {
    return node?.type === NODE_TYPE || node?.comfyClass === NODE_TYPE;
}

function hideWidget(entry) {
    if (!entry) return;
    entry.hidden = true;
    entry.draw = () => {};
    entry.computeSize = () => [0, -4];
}

function setWidgetValue(entry, value) {
    if (!entry) return;
    entry.value = value;
    entry.callback?.(value);
}

function parseState(value) {
    try {
        const parsed = JSON.parse(value || "{}");
        return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
        return {};
    }
}

function fnv1a(...parts) {
    let value = 0x811c9dc5;
    for (const byte of encoder.encode(parts.join("\x1f"))) {
        value ^= byte;
        value = Math.imul(value, 0x01000193) >>> 0;
    }
    return value;
}

function resolvePattern(pattern, seed, subjectKey, reroll) {
    let group = 0;
    return String(pattern).replace(/\{([^{}]*)\}/g, (_match, choices) => {
        const options = choices.split("|");
        const selected = options[fnv1a(seed, subjectKey, reroll, group) % options.length] || "";
        group += 1;
        return selected;
    });
}

function tokenCount(text) {
    return (String(text).match(TOKEN_RE) || []).length;
}

function isNegative(category) {
    return category.toLocaleLowerCase().includes("negative prompt");
}

function formatWeight(text, weight) {
    return weight === 1 ? text : `(${text}:${Number(weight).toFixed(2).replace(/0+$/, "").replace(/\.$/, "")})`;
}

function categoryTone(category) {
    if (category === "Positive Prompts") return "positive";
    if (category === "Negative Prompts") return "negative";
    if (category.startsWith("Character")) return "character";
    if (category === "Wardrobe") return "wardrobe";
    if (category === "Scene") return "scene";
    if (category === "Composition & Pose") return "composition";
    if (category === "Body Visibility") return "body";
    if (category === "Art Style") return "art";
    if (category.startsWith("Species")) return "species";
    return "nsfw";
}

function categoryIcon(category) {
    return {
        character: "◉", wardrobe: "◈", scene: "◇", composition: "⌁", body: "◎",
        art: "✦", species: "♧", nsfw: "⚠", positive: "✓", negative: "✕",
    }[categoryTone(category)];
}

function injectStyles() {
    if (document.getElementById("dasiwa-wildcard-builder-style")) return;
    const style = document.createElement("style");
    style.id = "dasiwa-wildcard-builder-style";
    style.textContent = `
.dasiwa-wildcard-builder { box-sizing:border-box; display:flex; flex-direction:column; gap:7px; width:100%; min-width:0; height:100%; overflow:hidden; color:var(--input-text,#ddd); font:12px sans-serif; }
.dasiwa-wildcard-controls { flex:none; min-width:0; }
.dasiwa-wildcard-selections { flex:1 1 auto; min-height:0; overflow:auto; padding-right:3px; }
.dasiwa-wildcard-preview-segment { flex:0 1 150px; min-height:72px; overflow:auto; padding-right:3px; }
.dasiwa-wildcard-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:7px; margin:0; }
.dasiwa-wildcard-regenerate { border:1px solid #5794c7; border-radius:4px; background:#18334c; color:#d9efff; padding:4px 9px; cursor:pointer; }
.dasiwa-wildcard-clear { border:1px solid #626d76; border-radius:4px; background:#242b30; color:#d9e1e7; padding:4px 9px; cursor:pointer; }
.dasiwa-wildcard-auto-reroll { display:flex; align-items:center; gap:3px; color:#c6d2db; cursor:pointer; }
.dasiwa-wildcard-badge { margin-left:auto; border-radius:10px; background:#2b3946; padding:4px 8px; font-size:11px; white-space:nowrap; }
.dasiwa-wildcard-library-heading { display:block; width:100%; margin:10px 0 4px; padding:4px 6px; border:0; border-bottom:1px solid #40515e; background:transparent; color:#d6e7f5; font-weight:700; letter-spacing:.04em; text-align:left; text-transform:uppercase; cursor:pointer; }
.dasiwa-wildcard-library-heading:hover { background:#263541; }
.dasiwa-wildcard-category { border:1px solid #344553; border-radius:4px; margin:4px 0; background:#171d22; }
.dasiwa-wildcard-category > summary { display:flex; align-items:center; gap:7px; cursor:pointer; padding:5px 7px; font-weight:600; color:#d6e7f5; }
.dasiwa-wildcard-selected-count { margin-left:auto; border:1px solid #4d9d6b; border-radius:9px; background:#173322; color:#a9efbd; padding:1px 6px; font-size:10px; font-weight:700; white-space:nowrap; }
.dasiwa-wildcard-category[data-has-selection="true"] { border-color:#4d9d6b; }
.dasiwa-wildcard-category[data-tone="character"] > summary { color:#a9d4ff; background:#17263a; }
.dasiwa-wildcard-category[data-tone="wardrobe"] > summary { color:#dbb7ff; background:#2b1c3b; }
.dasiwa-wildcard-category[data-tone="scene"] > summary { color:#9ce3e3; background:#163436; }
.dasiwa-wildcard-category[data-tone="composition"] > summary { color:#f0cf8f; background:#392d16; }
.dasiwa-wildcard-category[data-tone="body"] > summary { color:#f0b88a; background:#392417; }
.dasiwa-wildcard-category[data-tone="art"] > summary { color:#f1a9e1; background:#391c34; }
.dasiwa-wildcard-category[data-tone="species"] > summary { color:#aee4a8; background:#1b351c; }
.dasiwa-wildcard-category[data-tone="nsfw"] > summary { color:#edb0d8; background:#38172d; }
.dasiwa-wildcard-category[data-tone="positive"] > summary { color:#9ee6ae; background:#152a1b; }
.dasiwa-wildcard-category[data-tone="negative"] > summary { color:#ffb2b2; background:#32191d; }
.dasiwa-wildcard-columns { display:grid; grid-template-columns:18px minmax(70px,.8fr) 58px minmax(0,1.5fr); gap:6px; padding:3px 7px; color:#8897a4; font-size:10px; text-transform:uppercase; letter-spacing:.04em; }
.dasiwa-wildcard-subject { display:grid; grid-template-columns:18px minmax(70px,.8fr) 58px minmax(0,1.5fr); min-width:0; align-items:start; gap:6px; padding:5px 7px; border-top:1px solid #29343d; }
.dasiwa-wildcard-subject label { line-height:20px; overflow-wrap:anywhere; cursor:pointer; }
.dasiwa-wildcard-weight { width:56px; box-sizing:border-box; background:#101519; border:1px solid #4a5d6c; color:inherit; border-radius:3px; padding:2px; }
.dasiwa-wildcard-preview { min-width:0; color:#9cbed4; line-height:18px; overflow-wrap:anywhere; white-space:normal; }
.dasiwa-wildcard-subject:not(.enabled) .dasiwa-wildcard-preview { color:#62717c; }
.dasiwa-wildcard-output-preview { box-sizing:border-box; min-height:100%; margin:0; border:1px solid #40515e; border-radius:4px; background:#11171b; padding:7px; }
.dasiwa-wildcard-output-preview h4 { margin:0 0 4px; font-size:11px; color:#d6e7f5; }
.dasiwa-wildcard-output-preview p { margin:0 0 7px; color:#b9d4e8; line-height:18px; overflow-wrap:anywhere; white-space:normal; }
.dasiwa-wildcard-output-preview p:last-child { margin-bottom:0; }
.dasiwa-wildcard-output-preview .negative { color:#edafb3; }
`;
    document.head.appendChild(style);
}

async function loadLibrary() {
    const response = await fetch(LIBRARY_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`library request failed (${response.status})`);
    const payload = await response.json();
    if (!payload.categories) throw new Error(payload.error || "invalid wildcard library response");
    return payload.categories;
}

function entryValues(subject, style, entryType) {
    const source = style === "Natural Language" ? "nl" : "booru";
    return subject[`${source}_${entryType}`] || [];
}

function randomIndex(limit) {
    if (limit <= 1) return 0;
    const range = 0x100000000;
    const ceiling = range - (range % limit);
    const value = new Uint32Array(1);
    do {
        crypto.getRandomValues(value);
    } while (value[0] >= ceiling);
    return value[0] % limit;
}

function subjectText(category, subject, entryType, state, style, seed, reroll) {
    const key = `${category}/${subject.subject}/${entryType}`;
    const config = state[key] || {};
    const text = entryValues(subject, style, entryType)
        .filter(Boolean)
        .map((entry) => resolvePattern(entry, seed, key, reroll))
        .filter(Boolean)
        .join(", ");
    return { key, text, weight: Number.isFinite(Number(config.weight)) ? Number(config.weight) : 1, enabled: !!config.enabled };
}

function assemble(categories, state, style, seed, reroll, budget, negative) {
    const items = [];
    for (const [category, subjects] of Object.entries(categories)) {
        if (isNegative(category) !== negative) continue;
        for (const subject of subjects) {
            for (const entryType of ["wildcards", "presets"]) {
                const result = subjectText(category, subject, entryType, state, style, seed, reroll);
                if (result.enabled && result.text) items.push({ ...result, index: items.length });
            }
        }
    }
    while (items.length && tokenCount(items.map((item) => formatWeight(item.text, item.weight)).join(", ")) > budget) {
        let remove = 0;
        for (let i = 1; i < items.length; i++) {
            if (items[i].weight < items[remove].weight || (items[i].weight === items[remove].weight && items[i].index > items[remove].index)) remove = i;
        }
        items.splice(remove, 1);
    }
    const prompt = items.map((item) => formatWeight(item.text, item.weight)).join(", ");
    return { prompt, tokens: tokenCount(prompt) };
}

app.registerExtension({
    name: "DaSiWa.WildcardPresetPromptBuilder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        installWildcardPicker = function ({ fitInitialSize = false } = {}) {
            if (this.dasiwaWildcardPickerInstalled) return;
            this.dasiwaWildcardPickerInstalled = true;
            injectStyles();
            const stateWidget = widget(this, "selection_state");
            const rerollWidget = widget(this, "reroll");
            const autoRerollWidget = widget(this, "auto_reroll");
            hideWidget(stateWidget);
            hideWidget(rerollWidget);
            hideWidget(autoRerollWidget);
            const state = parseState(stateWidget?.value);
            const expanded = new Set();
            const container = document.createElement("div");
            container.className = "dasiwa-wildcard-builder";
            container.textContent = "Loading wildcard library…";
            this.addDOMWidget("wildcard_subjects", "custom", container, {
                serialize: false,
                hideOnZoom: false,
                // Do not derive this from node.size: ComfyUI uses getHeight()
                // to calculate node.size, so that feedback loop grows the node
                // on every layout pass.
                getHeight: () => PICKER_HEIGHT,
            });
            const sync = () => {
                setWidgetValue(stateWidget, JSON.stringify(state));
                this.properties = this.properties || {};
                this.properties.wildcard_selection_state = stateWidget?.value || "{}";
                this.setDirtyCanvas?.(true, true);
            };
            const values = () => ({
                style: widget(this, "style")?.value || "Booru",
                seed: Number(widget(this, "seed")?.value || 0),
                reroll: Number(rerollWidget?.value || 0),
                budget: Math.max(1, Number(widget(this, "token_budget")?.value || 200)),
            });
            let categories = null;

            const renderRow = (details, category, subject, entryType, settings) => {
                const item = subjectText(category, subject, entryType, state, settings.style, settings.seed, settings.reroll);
                if (!item.text) return;
                const row = document.createElement("div");
                row.className = `dasiwa-wildcard-subject${item.enabled ? " enabled" : ""}`;
                const enabled = document.createElement("input");
                const inputId = `dasiwa-wildcard-${fnv1a(item.key)}`;
                enabled.id = inputId;
                enabled.type = "checkbox";
                enabled.checked = item.enabled;
                enabled.onchange = () => { state[item.key] = { enabled: enabled.checked, weight: item.weight }; sync(); render(); };
                const label = document.createElement("label");
                label.htmlFor = inputId;
                label.textContent = subject.subject;
                label.title = subject.subject;
                const weight = document.createElement("input");
                weight.className = "dasiwa-wildcard-weight";
                weight.type = "number";
                weight.min = "-10";
                weight.max = "10";
                weight.step = "0.05";
                weight.value = item.weight;
                weight.onchange = () => { state[item.key] = { enabled: item.enabled, weight: Number(weight.value) || 1 }; sync(); render(); };
                const preview = document.createElement("span");
                preview.className = "dasiwa-wildcard-preview";
                preview.textContent = item.text;
                row.append(enabled, label, weight, preview);
                details.appendChild(row);
            };

            const renderSection = (entryType, settings, target) => {
                const sectionCategoryKeys = Object.entries(categories)
                    .filter(([, subjects]) => subjects.some((subject) => entryValues(subject, settings.style, entryType).some(Boolean)))
                    .map(([category]) => `${entryType}/${category}`);
                const sectionIsOpen = sectionCategoryKeys.some((key) => expanded.has(key));
                const heading = document.createElement("button");
                heading.className = "dasiwa-wildcard-library-heading";
                heading.type = "button";
                heading.textContent = `${entryType === "wildcards" ? "Wildcards" : "Presets"} ${sectionIsOpen ? "▾" : "▸"}`;
                heading.title = sectionIsOpen ? `Collapse every ${entryType} category` : `Expand every ${entryType} category`;
                heading.onclick = () => {
                    for (const key of sectionCategoryKeys) {
                        if (sectionIsOpen) expanded.delete(key);
                        else expanded.add(key);
                    }
                    render();
                };
                target.appendChild(heading);
                for (const [category, subjects] of Object.entries(categories)) {
                    if (!subjects.some((subject) => entryValues(subject, settings.style, entryType).some(Boolean))) continue;
                    const details = document.createElement("details");
                    details.className = "dasiwa-wildcard-category";
                    details.dataset.tone = categoryTone(category);
                    const selectedCount = subjects.filter((subject) => {
                        const key = `${category}/${subject.subject}/${entryType}`;
                        return entryValues(subject, settings.style, entryType).some(Boolean) && state[key]?.enabled;
                    }).length;
                    details.dataset.hasSelection = String(selectedCount > 0);
                    const detailKey = `${entryType}/${category}`;
                    details.open = expanded.has(detailKey);
                    details.ontoggle = () => {
                        details.open ? expanded.add(detailKey) : expanded.delete(detailKey);
                    };
                    const categorySummary = document.createElement("summary");
                    const categoryLabel = document.createElement("span");
                    categoryLabel.textContent = `${categoryIcon(category)}  ${category}`;
                    categorySummary.appendChild(categoryLabel);
                    if (selectedCount) {
                        const selectedBadge = document.createElement("span");
                        selectedBadge.className = "dasiwa-wildcard-selected-count";
                        selectedBadge.textContent = `✓ ${selectedCount} selected`;
                        categorySummary.appendChild(selectedBadge);
                    }
                    details.appendChild(categorySummary);
                    const columns = document.createElement("div");
                    columns.className = "dasiwa-wildcard-columns";
                    columns.append(document.createElement("span"), ...["Subject", "Weight", "Current Pick"].map((text) => {
                        const heading = document.createElement("span");
                        heading.textContent = text;
                        return heading;
                    }));
                    details.appendChild(columns);
                    for (const subject of subjects) renderRow(details, category, subject, entryType, settings);
                    target.appendChild(details);
                }
            };

            const render = () => {
                if (!categories) return;
                const settings = values();
                const positive = assemble(categories, state, settings.style, settings.seed, settings.reroll, settings.budget, false);
                const negative = assemble(categories, state, settings.style, settings.seed, settings.reroll, settings.budget, true);
                container.replaceChildren();
                const controls = document.createElement("div");
                controls.className = "dasiwa-wildcard-controls";
                const selections = document.createElement("div");
                selections.className = "dasiwa-wildcard-selections";
                const previewSegment = document.createElement("div");
                previewSegment.className = "dasiwa-wildcard-preview-segment";
                const toolbar = document.createElement("div");
                toolbar.className = "dasiwa-wildcard-toolbar";
                const regenerate = document.createElement("button");
                regenerate.className = "dasiwa-wildcard-regenerate";
                regenerate.type = "button";
                regenerate.textContent = "🎲 New Picks";
                regenerate.title = "Resolve every {A|B|C} choice again while keeping the current seed.";
                regenerate.onclick = () => { setWidgetValue(rerollWidget, settings.reroll + 1); sync(); render(); };
                const randomSelect = document.createElement("button");
                randomSelect.className = "dasiwa-wildcard-regenerate";
                randomSelect.type = "button";
                randomSelect.textContent = "🎲 Random Select";
                randomSelect.title = "Replace the current selection with 1–10 random available Presets and Wildcards.";
                randomSelect.onclick = () => {
                    const candidates = [];
                    for (const [category, subjects] of Object.entries(categories)) {
                        for (const subject of subjects) {
                            for (const entryType of ["presets", "wildcards"]) {
                                if (entryValues(subject, settings.style, entryType).some(Boolean)) {
                                    candidates.push(`${category}/${subject.subject}/${entryType}`);
                                }
                            }
                        }
                    }
                    for (const config of Object.values(state)) {
                        if (config && typeof config === "object") config.enabled = false;
                    }
                    const count = Math.min(candidates.length, 1 + randomIndex(10));
                    for (let selected = 0; selected < count; selected++) {
                        const index = selected + randomIndex(candidates.length - selected);
                        [candidates[selected], candidates[index]] = [candidates[index], candidates[selected]];
                        const key = candidates[selected];
                        state[key] = { enabled: true, weight: Number(state[key]?.weight) || 1 };
                    }
                    sync();
                    render();
                };
                const clear = document.createElement("button");
                clear.className = "dasiwa-wildcard-clear";
                clear.type = "button";
                clear.textContent = "Deselect All";
                clear.title = "Disable every selected wildcard and preset; saved weights remain unchanged.";
                clear.onclick = () => {
                    for (const config of Object.values(state)) {
                        if (config && typeof config === "object") config.enabled = false;
                    }
                    sync();
                    render();
                };
                const collapse = document.createElement("button");
                collapse.className = "dasiwa-wildcard-clear";
                collapse.type = "button";
                collapse.textContent = "Collapse All";
                collapse.title = "Collapse all Wildcards and Presets categories.";
                collapse.onclick = () => { expanded.clear(); render(); };
                const autoReroll = document.createElement("label");
                autoReroll.className = "dasiwa-wildcard-auto-reroll";
                const autoRerollInput = document.createElement("input");
                autoRerollInput.type = "checkbox";
                autoRerollInput.checked = Boolean(autoRerollWidget?.value);
                autoRerollInput.onchange = () => {
                    setWidgetValue(autoRerollWidget, autoRerollInput.checked);
                    this.setDirtyCanvas?.(true, true);
                };
                autoReroll.append(autoRerollInput, " New picks on every queue");
                const badge = document.createElement("span");
                badge.className = "dasiwa-wildcard-badge";
                badge.textContent = `+ ${positive.tokens}/${settings.budget} · − ${negative.tokens}/${settings.budget}`;
                toolbar.append(regenerate, randomSelect, clear, collapse, autoReroll, badge);
                controls.appendChild(toolbar);
                renderSection("presets", settings, selections);
                renderSection("wildcards", settings, selections);
                const output = document.createElement("div");
                output.className = "dasiwa-wildcard-output-preview";
                const positiveHeading = document.createElement("h4");
                positiveHeading.textContent = "Complete Prompt Preview";
                const positivePreview = document.createElement("p");
                positivePreview.textContent = positive.prompt || "—";
                const negativeHeading = document.createElement("h4");
                negativeHeading.textContent = "Negative Prompt Preview";
                const negativePreview = document.createElement("p");
                negativePreview.className = "negative";
                negativePreview.textContent = negative.prompt || "—";
                output.append(positiveHeading, positivePreview, negativeHeading, negativePreview);
                previewSegment.appendChild(output);
                container.append(controls, selections, previewSegment);
                this.setDirtyCanvas?.(true, true);
            };

            // Installation can precede restoration of saved widget values.
            // Reload without syncing: the serialized widget remains canonical.
            this.dasiwaWildcardReloadState = () => {
                for (const key of Object.keys(state)) delete state[key];
                Object.assign(state, parseState(stateWidget?.value || this.properties?.wildcard_selection_state));
                render();
            };

            for (const name of ["style", "seed", "token_budget"]) {
                const control = widget(this, name);
                if (!control) continue;
                const callback = control.callback;
                control.callback = (...args) => { callback?.apply(control, args); render(); };
            }
            loadLibrary().then((library) => {
                categories = library;
                render();
            }).catch((error) => {
                container.textContent = `Wildcard library unavailable: ${error.message}`;
                console.error("[DaSiWa Wildcard Builder]", error);
            });

            // Only size a brand-new node once, after its DOM widget exists.
            // loadedGraphNode() runs before this frame for restored workflows,
            // so their saved user-controlled dimensions are never overwritten.
            if (fitInitialSize) {
                requestAnimationFrame(() => {
                    if (this.dasiwaWildcardPickerRestored) return;
                    this.setSize([
                        Math.max(this.size?.[0] || 0, 620),
                        Math.max(this.size?.[1] || 0, 620),
                    ]);
                    this.setDirtyCanvas?.(true, true);
                });
            }
        };

        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installWildcardPicker.call(this, { fitInitialSize: true });
            return result;
        };

        // LiteGraph restores widgets_values before invoking onConfigure().
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            this.dasiwaWildcardReloadState?.();
            return result;
        };

        return undefined;
    },

    loadedGraphNode(node) {
        if (!isWildcardNode(node)) return;
        node.dasiwaWildcardPickerRestored = true;
        // ComfyUI restores saved nodes without guaranteeing onNodeCreated()
        // will run after this extension has registered. Rebuild the transient,
        // non-serialized DOM widget in that lifecycle path as well.
        installWildcardPicker?.call(node);
        node.dasiwaWildcardReloadState?.();
    },

    afterConfigureGraph() {
        // On a browser refresh, ComfyUI can restore the graph before the DOM
        // widget layer has mounted. Revisit every restored instance after graph
        // configuration so the picker is always registered with that layer.
        for (const node of app.graph?._nodes || []) {
            if (!isWildcardNode(node)) continue;
            installWildcardPicker?.call(node);
            node.dasiwaWildcardReloadState?.();
        }
    },
});
