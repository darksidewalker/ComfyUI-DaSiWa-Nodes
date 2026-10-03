const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { TextEncoder } = require('node:util');

const source = fs.readFileSync(path.join(__dirname, '..', 'js', 'wildcard_preset_prompt_builder.js'), 'utf8')
    .replace(/^import .*;\s*$/gm, '');
const NODE_TYPE = 'DaSiWa_WildcardPresetPromptBuilder';
const forest = 'Scene/Forest/wildcards';
const beach = 'Scene/Beach/presets';
const saved = JSON.stringify({ [forest]: { enabled: true, weight: 1.5 } });

class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; }
    append(...items) { this.children.push(...items); }
    appendChild(item) { this.append(item); }
    replaceChildren(...items) { this.children = items; }
}
function descendants(root) {
    return root && typeof root === 'object' ? [root, ...(root.children || []).flatMap(descendants)] : [];
}
function checkboxes(node) {
    return descendants(node.container).filter(entry => entry.tag === 'input' && entry.type === 'checkbox' && entry.id);
}

async function setup() {
    let extension;
    let resolveLibrary;
    const library = new Promise(resolve => { resolveLibrary = resolve; });
    const app = { registerExtension: value => { extension = value; }, graph: { _nodes: [] } };
    vm.runInNewContext(source, {
        app, TextEncoder, console,
        document: { getElementById: () => null, createElement: tag => new Element(tag), head: new Element('head') },
        fetch: async () => ({ ok: true, json: () => library }),
        requestAnimationFrame: () => {},
    });
    class Node {
        constructor() {
            this.type = NODE_TYPE;
            this.properties = {};
            this.widgets = [{ name: 'selection_state', value: '{}' }, { name: 'style', value: 'Booru' },
                { name: 'token_budget', value: 200 }, { name: 'reroll', value: 0 }];
            this.domCount = 0;
        }
        onNodeCreated() { return 'created'; }
        onConfigure(value) { this.configureArgument = value; return 'configured'; }
        addDOMWidget(name, type, element) { this.container = element; this.domCount += 1; }
    }
    await extension.beforeRegisterNodeDef(Node, { name: NODE_TYPE });
    return { Node, extension, app, async loadLibrary() {
        resolveLibrary({ categories: { Scene: [
            { subject: 'Forest', booru_wildcards: ['forest'], booru_presets: [] },
            { subject: 'Beach', booru_wildcards: [], booru_presets: ['beach'] },
        ] } });
        await new Promise(resolve => setImmediate(resolve));
    } };
}

(async () => {
    // Exercise each restoration hook independently, both before and after the async library response.
    for (const hook of ['onConfigure', 'loadedGraphNode', 'afterConfigureGraph']) {
        for (const loadFirst of [false, true]) {
            const environment = await setup();
            const node = new environment.Node();
            assert.equal(node.onNodeCreated(), 'created');
            if (loadFirst) await environment.loadLibrary();
            node.widgets[0].value = saved;
            node.properties.wildcard_selection_state = saved;
            if (hook === 'onConfigure') {
                assert.equal(node.onConfigure('argument'), 'configured');
                assert.equal(node.configureArgument, 'argument');
            } else if (hook === 'loadedGraphNode') environment.extension.loadedGraphNode(node);
            else {
                environment.app.graph._nodes = [node, { type: 'OtherNode' }];
                environment.extension.afterConfigureGraph();
            }
            if (!loadFirst) await environment.loadLibrary();
            assert.equal(node.widgets[0].value, saved, 'restore must not rewrite serialized state');
            const forestInput = checkboxes(node).find(entry => entry.checked);
            assert.ok(forestInput, `${hook}: restored wildcard is not checked (loadFirst=${loadFirst})`);
            assert.ok(descendants(node.container).some(entry => entry.className === 'dasiwa-wildcard-weight' && entry.value === 1.5));
            const beachInput = checkboxes(node).find(entry => !entry.checked);
            beachInput.checked = true;
            beachInput.onchange();
            assert.deepEqual(JSON.parse(node.widgets[0].value), {
                [forest]: { enabled: true, weight: 1.5 }, [beach]: { enabled: true, weight: 1 },
            });
            const newPicks = descendants(node.container).find(entry => entry.textContent === '🎲 New Picks');
            newPicks.onclick();
            assert.equal(JSON.parse(node.widgets[0].value)[forest].weight, 1.5);
            assert.equal(node.widgets.find(entry => entry.name === 'reroll').value, 1);
            // Reconfigure the same installed picker: replace, do not merge with previous selections.
            node.widgets[0].value = '{}';
            node.properties.wildcard_selection_state = saved;
            node.onConfigure();
            assert.equal(checkboxes(node).some(entry => entry.checked), false);
            assert.equal(node.widgets[0].value, '{}');
            environment.extension.loadedGraphNode(node);
            assert.equal(node.domCount, 1, 'restore must not install duplicate DOM widgets');
            console.log(`PASS: ${hook}, library ${loadFirst ? 'before' : 'after'} restore`);
        }
    }
    // Nodes reached without onNodeCreated still receive an initialized picker.
    const environment = await setup();
    const node = new environment.Node();
    node.widgets[0].value = saved;
    environment.extension.loadedGraphNode(node);
    await environment.loadLibrary();
    assert.equal(checkboxes(node).some(entry => entry.checked), true);
    assert.equal(node.domCount, 1);
    console.log('PASS: late picker installation');
})().catch(error => { console.error(error); process.exitCode = 1; });
