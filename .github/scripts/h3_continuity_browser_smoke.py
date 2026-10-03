"""Real ComfyUI graph restore/save smoke test; never queues a generation.

Run against an isolated server loading THIS checkout:
  python .github/scripts/h3_continuity_browser_smoke.py http://127.0.0.1:8199
Test-only dependency: playwright (+ its Chromium install). No runtime dependency.
"""
import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen


def main():
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--asset-prefix", default="/extensions/ComfyUI-DaSiWa-Nodes")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    root = Path(__file__).resolve().parents[2]
    for name in ("minimax_h3_director.js", "minimax_h3_continuity.js", "minimax_h3_forge_state.js"):
        with urlopen(url + args.asset_prefix + "/" + name, timeout=15) as response:
            assert response.read() == (root / "js" / name).read_bytes(), f"Wrong served asset: {name}"

    with sync_playwright() as p:
        options = {"headless": True, "args": ["--no-sandbox"]}
        if os.environ.get("H3_TEST_CHROME"):
            options["executable_path"] = os.environ["H3_TEST_CHROME"]
        browser = p.chromium.launch(**options)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="networkidle")
        page.wait_for_function("!!window.LiteGraph?.registered_node_types['MiniMaxH3Director']", timeout=60000)
        page.evaluate("async () => { window.testApp = (await import('/scripts/app.js')).app; }")
        page.evaluate("""() => {
            const ext = testApp.extensions.find(e => e.name === 'DaSiWa.MiniMaxH3Director');
            window.h3Hooks = {created: 0, loaded: 0};
            for (const [key, counter] of [['nodeCreated','created'], ['loadedGraphNode','loaded']]) {
                const old = ext[key];
                ext[key] = function(node, ...args) {
                    if (node.comfyClass === 'MiniMaxH3Director') h3Hooks[counter]++;
                    return old.call(this, node, ...args);
                };
            }
        }""")
        results = []
        for vue in (False, True):
            page.evaluate("vue => testApp.ui.settings.setSettingValue('Comfy.VueNodes.Enabled', vue)", vue)
            for version, frames, active in ((2, 17, True), (2, 238, True), (2, 357, True), (1, 119, False)):
                expected = frames / 24 if active else 10
                page.evaluate("""({version, frames, active}) => {
                    testApp.graph.clear();
                    const node = LiteGraph.createNode('MiniMaxH3Director');
                    testApp.graph.add(node);
                    const saved = testApp.graph.serialize();
                    const serial = saved.nodes.find(n => String(n.id) === String(node.id));
                    const widgets = node.widgets.filter(w => w.serialize !== false);
                    const values = {
                        duration: 10,
                        timeline_data: JSON.stringify({version: 1, items: [], prompt_blocks: [],
                            continuity: {version, extension_frames: frames, capture: false,
                                source_id: 'test-parent', operation: active ? 'continue' : 'new',
                                continuation_prompt: 'Turn left.', idea: 'Camera follows.'}})
                    };
                    for (const [name, value] of Object.entries(values)) {
                        serial.widgets_values[widgets.findIndex(w => w.name === name)] = value;
                        if (serial.widgets_values_named) serial.widgets_values_named[name] = value;
                    }
                    window.h3Fixture = saved;
                    window.h3Hooks.created = window.h3Hooks.loaded = 0;
                }""", {"version": version, "frames": frames, "active": active})
                page.evaluate("async () => { await testApp.loadGraphData(h3Fixture); }")
                page.wait_for_function("""expected => {
                    const n = testApp.graph._nodes.find(n => n.comfyClass === 'MiniMaxH3Director');
                    return n?.widgets.find(w => w.name === 'duration').value === expected &&
                        JSON.parse(n.widgets.find(w => w.name === 'timeline_data').value).continuity.version === 3;
                }""", arg=expected, timeout=15000)
                page.locator(".ds-h3-root").wait_for(state="visible")
                assert page.locator(".lg-node:has(.ds-h3-root)").count() == int(vue)
                initial = page.evaluate("""() => {
                    const n = testApp.graph._nodes.find(n => n.comfyClass === 'MiniMaxH3Director');
                    const c = JSON.parse(n.widgets.find(w => w.name === 'timeline_data').value).continuity;
                    return {c, hooks: {...h3Hooks}, roots: document.querySelectorAll('.ds-h3-root').length};
                }""")
                assert initial["hooks"]["created"] > 0 and initial["hooks"]["loaded"] > 0, initial
                assert initial["roots"] == 1, initial
                assert initial["c"]["operation"] == ("continue" if active else "new"), initial
                assert initial["c"]["capture"] is False, initial
                assert initial["c"]["continuation_prompt"] == "Turn left.\nCamera follows.", initial
                assert "extension_frames" not in initial["c"], initial
                # Repeated loadedGraphNode must not install a second widget/timer or remigrate.
                page.evaluate("""() => {
                    const n = testApp.graph._nodes.find(n => n.comfyClass === 'MiniMaxH3Director');
                    const ext = testApp.extensions.find(e => e.name === 'DaSiWa.MiniMaxH3Director');
                    ext.loadedGraphNode(n); ext.loadedGraphNode(n);
                    const w = n.widgets.find(w => w.name === 'duration');
                    w.value = 7.25; w.callback?.(w.value);
                    const fps = n.widgets.find(w => w.name === 'frame_rate');
                    fps.value = 24; fps.callback?.(fps.value);
                    window.h3RoundTrip = testApp.graph.serialize();
                }""")
                page.evaluate("async () => { await testApp.loadGraphData(h3RoundTrip); }")
                page.wait_for_function("""() => {
                    const n = testApp.graph._nodes.find(n => n.comfyClass === 'MiniMaxH3Director');
                    return n?.widgets.find(w => w.name === 'duration').value === 7.25 &&
                        n.__dasiwaH3State?.().continuity.version === 3;
                }""", timeout=15000)
                restored = page.evaluate("""() => {
                    const n = testApp.graph._nodes.find(n => n.comfyClass === 'MiniMaxH3Director');
                    return {c: n.__dasiwaH3State().continuity,
                        domWidgets: n.widgets.filter(w => w.name === 'minimax_h3_director_ui').length,
                        builderDuration: JSON.parse(n.widgets.find(w => w.name === 'builder_state').value).duration};
                }""")
                assert restored["c"] == initial["c"], restored
                assert restored["domWidgets"] == 1, restored
                assert restored["builderDuration"] == 7.25, restored
                results.append({"vue": vue, "legacy_version": version, "frames": frames,
                                "active": active, "migrated_duration": expected, "save_reload": "passed"})
        page.evaluate("() => testApp.graph.clear()")
        assert not errors, errors
        browser.close()
        print(json.dumps({"cases": results, "uncaught_browser_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
