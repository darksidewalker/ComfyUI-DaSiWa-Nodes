"""Exercise the real extension in an isolated browser; never call live /free."""
import json
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")


def test_memory_menu_requests_and_lifecycle():
    source_path = Path(__file__).resolve().parents[1] / "js" / "dasiwa_free_memory.js"
    source = "\n".join(line for line in source_path.read_text().splitlines() if not line.startswith("import "))
    source = source.replace("import.meta.url", json.dumps(source_path.as_uri()))
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 800, "height": 600})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.set_content('<div data-testid="legacy-topbar-container"><div class="flex"></div></div>')
        page.add_script_tag(content="""
            window.requests = [];
            window.alerts = [];
            window.alert = value => alerts.push(value);
            window.failRequest = false;
            window.app = {
                ui: { settings: {
                    addSetting(value) { window.setting = value; },
                    getSettingValue() { return true; }
                } },
                registerExtension(value) { window.extension = value; }
            };
            window.api = { async fetchApi(url, options) {
                requests.push({url, method: options.method, body: JSON.parse(options.body)});
                return {ok: !failRequest, status: failRequest ? 500 : 200};
            } };
        """)
        page.add_script_tag(content=source)
        page.evaluate("extension.init(); extension.setup()")
        menu = page.locator("#dasiwa-free-memory-menu")
        toggle = page.get_by_role("button", name="Free memory", exact=True)
        for index, (label, free_memory) in enumerate([
            ("Free VRAM", False), ("Free System RAM", True), ("Free Both", True)
        ]):
            toggle.click()
            assert menu.is_visible()
            choice = page.get_by_role("button", name=label, exact=True)
            data_free = choice.get_attribute("data-free")
            choice.click()
            page.wait_for_function("count => requests.length === count", arg=index + 1)
            assert not menu.is_visible()
            assert menu.locator(f'button[data-free="{data_free}"]').is_enabled()
            assert page.evaluate("requests.at(-1)") == {
                "url": "/free", "method": "POST",
                "body": {"unload_models": True, "free_memory": free_memory}
            }
        toggle.click()
        page.mouse.click(700, 400)
        assert not menu.is_visible()
        page.evaluate("failRequest = true")
        toggle.click()
        page.get_by_role("button", name="Free Both", exact=True).click()
        page.wait_for_function("alerts.length === 1")
        assert page.evaluate("alerts[0]") == "Could not free memory: HTTP 500"
        assert menu.locator('button[data-free="both"]').is_enabled()
        page.evaluate("setting.onChange(false)")
        assert page.locator("#dasiwa-free-memory").count() == 0
        assert menu.count() == 0
        page.evaluate("setting.onChange(true)")
        assert page.locator("#dasiwa-free-memory").count() == 1
        toggle.click()
        assert page.get_by_role("button", name="Free Both", exact=True).is_visible()
        assert not errors
        browser.close()
