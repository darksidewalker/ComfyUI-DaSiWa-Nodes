"""Seed Control manual-entry regression against an isolated ComfyUI instance.

uv run --no-project --with playwright python tools/verify_seed_control.py --url http://127.0.0.1:8199
Uses a fresh browser context and never queues model execution.
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", required=True)
args = parser.parse_args()

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(args.url, wait_until="domcontentloaded")
    page.wait_for_function("Boolean(window.comfyAPI?.app?.app?.graph && "
                           "window.LiteGraph?.registered_node_types?.DaSiWa_SeedControl)", timeout=90000)
    page.evaluate("()=>{window.app=window.comfyAPI.app.app}")
    served = page.request.get(args.url.rstrip("/") + "/extensions/ComfyUI-DaSiWa-Nodes/dasiwa_seed_control.js")
    assert served.ok and served.text() == (Path(__file__).parents[1] / "js/dasiwa_seed_control.js").read_text()
    for close in page.get_by_role("button", name="Close dialog", exact=True).all():
        close.click()

    for modern in [False, True]:
        page.evaluate("mode=>app.ui.settings.setSettingValue('Comfy.VueNodes.Enabled',mode)", modern)
        page.wait_for_function("mode=>LiteGraph.vueNodesMode===mode", arg=modern)
        for commit in ["blur", "Enter"]:
            page.evaluate("""()=>{app.graph.clear();window.n=LiteGraph.createNode('DaSiWa_SeedControl');
                n.pos=[100,100];app.graph.add(n)}""")
            root = page.locator(".ds-seed")
            field = root.locator(".ds-seed-num")
            random = root.get_by_role("button", name="Random", exact=True)
            fixed = root.get_by_role("button", name="Fixed", exact=True)
            expect(random).to_have_class("ds-seed-switch-seg active")
            last = root.get_by_role("button", name="Use Last", exact=True)
            expect(last).to_be_disabled()
            page.evaluate("window.seedInput=document.querySelector('.ds-seed-num')")
            field.fill("12345")
            if commit == "Enter":
                field.press("Enter")
            else:
                root.locator("summary").click()
            # ComfyUI versions differ on whether Enter itself blurs a text widget.
            field.blur()
            state = json.loads(page.evaluate("n.widgets.find(w=>w.name==='seed_control_state').value"))
            assert state["mode"] == "fixed" and state["last_seed"] == "12345", state
            expect(fixed).to_have_class("ds-seed-switch-seg active")
            expect(random).to_have_class("ds-seed-switch-seg")
            expect(last).to_be_enabled()
            assert page.evaluate("document.querySelector('.ds-seed-num')===window.seedInput")
            before = page.evaluate("n.widgets.find(w=>w.name==='seed_value').value")
            page.evaluate("()=>app.graphToPrompt()")
            assert str(page.evaluate("n.widgets.find(w=>w.name==='seed_value').value")) == str(before) == "12345"
            assert page.evaluate("document.querySelector('.ds-seed-num')===window.seedInput")
            saved = page.evaluate("app.graph.serialize()")
            page.evaluate("async saved=>{await app.loadGraphData(saved);window.n=app.graph._nodes[0]}", saved)
            expect(fixed).to_have_class("ds-seed-switch-seg active")
            expect(field).to_have_value("12345")
            random.click()
            expect(random).to_have_class("ds-seed-switch-seg active")
            field.fill("18446744073709551616")
            field.blur()
            expect(field).to_have_value("12345")
            expect(random).to_have_class("ds-seed-switch-seg active")
            print(f"PASS {'modern' if modern else 'classic'} / {commit}: fixed highlight, stable prompt seed, reload and overflow rejection")
    assert not errors, errors
    browser.close()
