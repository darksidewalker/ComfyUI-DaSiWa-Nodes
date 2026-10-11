"""Browser regression: uv run --no-project --with playwright python tests/test_h3_forge_resize.py"""
from pathlib import Path
import unittest

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class ForgeResizeTests(unittest.TestCase):
    def test_mixed_media_output_keeps_its_height_and_corner_resize(self):
        state = (ROOT / "js/minimax_h3_forge_state.js").read_text()
        state = "\n".join(line.removeprefix("export ") for line in state.splitlines())
        extension = (ROOT / "js/minimax_h3_forge.js").read_text()
        extension = "\n".join(line for line in extension.splitlines() if not line.startswith("import "))
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            for height in (900, 600):
                with self.subTest(viewport_height=height):
                    page = browser.new_page(viewport={"width": 1100, "height": height})
                    page.set_content("<!doctype html><html><head></head><body></body></html>")
                    page.evaluate(r"""() => {
                        window.app = {registerExtension() {}};
                        window.localStorage = {getItem: () => '{}', setItem() {}};
                        window.draft = ('A generated scene with image and audio references.\n').repeat(40);
                        window.rawFailure = false;
                        window.api = {apiURL: p => p, fetchApi: async url => {
                            if (url.endsWith('/models')) return {ok: true, json: async () => ({
                                models: [{id: 'local:test', label: 'Test'}], creativity: ['balanced'],
                                default_creativity: 'balanced', detail_levels: {}, default_detail: 5,
                                shot_counts: ['Auto', '3']
                            })};
                            if (window.rawFailure) return {ok: false, json: async () => ({raw: window.draft, message: 'Invalid draft'})};
                            return {ok: true, json: async () => ({mode: 'REF2VA', simple_prompt: window.draft,
                                fields: {}, model: 'local:test', unloaded: true, stats: {seconds: 1}})};
                        }};
                        const refs = [{kind: 'image', path: 'a.png', easy_role: 'character-1', role: 'subject',
                            instructions: '', item: {id: 'a'}}, {kind: 'audio', path: 'voice.wav', instructions: '', item: {id: 'b'}}];
                        window.forgeNode = {id: 1, properties: {}, graph: {setDirtyCanvas() {}}, __dasiwaH3Forge: {
                            references: () => refs, mode: () => 'REF2VA', contextKey: () => 'context',
                            duration: () => 5, updateReference: () => true, setStatus() {}, apply: () => true
                        }};
                    }""")
                    page.add_script_tag(content=state + "\n" + extension)
                    page.evaluate("window.DaSiWaH3Forge.open(window.forgeNode)")
                    page.locator(".ds-forge textarea").first.fill("A scene with a voice")
                    page.get_by_role("button", name="Generate", exact=True).click()
                    output = page.locator(".ds-forge pre")
                    self.assertEqual(output.inner_text(), page.evaluate("window.draft"))
                    styles = output.evaluate("e => ({resize: getComputedStyle(e).resize, height: e.getBoundingClientRect().height})")
                    self.assertEqual(styles["resize"], "vertical")
                    self.assertGreaterEqual(styles["height"], 140)
                    output.scroll_into_view_if_needed()
                    before = output.bounding_box()
                    page.mouse.move(before["x"] + before["width"] - 3, before["y"] + before["height"] - 3)
                    page.mouse.down()
                    page.mouse.move(before["x"] + before["width"] - 3, before["y"] + before["height"] + 67, steps=10)
                    page.mouse.up()
                    after = output.bounding_box()
                    self.assertGreater(after["height"], before["height"] + 40)
                    self.assertAlmostEqual(after["width"], before["width"], delta=1)
                    page.locator(".ds-forge .history button[title^='Show this prompt']").click()
                    self.assertAlmostEqual(output.bounding_box()["height"], after["height"], delta=1)
                    page.evaluate("window.rawFailure = true")
                    page.get_by_role("button", name="Regenerate", exact=True).click()
                    self.assertEqual(output.inner_text(), page.evaluate("window.draft"))
                    self.assertGreaterEqual(output.bounding_box()["height"], 140)
                    page.close()
            browser.close()


if __name__ == "__main__":
    unittest.main()
