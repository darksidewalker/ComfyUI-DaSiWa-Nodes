"""Chromium DOM regression checks with an explicit fixture API (no model inference).
Run: uv run --no-project --with playwright python .github/scripts/verify_h3_forge_ui.py
"""
import base64
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / "js/minimax_h3_forge.js").read_text()
state = base64.b64encode((ROOT / "js/minimax_h3_forge_state.js").read_bytes()).decode()
source = source.replace('import { app } from "../../scripts/app.js";', 'const app = {registerExtension() {}};')
source = source.replace('import { api } from "../../scripts/api.js";', 'const api = window.testApi;')
source = source.replace('"./minimax_h3_forge_state.js"', '"data:text/javascript;base64,' + state + '"')
fixture = """<!doctype html><body><script>
window.calls=[];
window.testApi={apiURL:()=>'',fetchApi:async(url,opts)=>{
  window.calls.push([url,JSON.parse(opts.body)]);
  return {ok:true,json:async()=>url.endsWith('/models') ? {
    models:[{id:'local:test',label:'Test fixture'}],creativity:['balanced'],default_creativity:'balanced',
    detail_levels:{5:'balanced'},default_detail:5,shot_counts:['Auto',1,2,3,4,5],default_shots:'Auto'
  } : {mode:'REF2VA',simple_prompt:'Fixture draft',fields:{ref:{}},easy:true,stats:{seconds:0},warnings:[],unloaded:true,model:'local:test'}};
}};
window.testNode={id:1,properties:window.restoredProperties || {},__dasiwaH3Forge:{mode:()=> 'REF2VA',items:()=>Array.from({length:9},(_,i)=>({id:'p'+i,type:'image',slot:i,value:'test.png'})),contextKey:()=> 'test-context',duration:()=>10,setStatus:()=>{},existingDefinitions:()=>({text:'',warning:''})}};
</script><script type="module">""" + source + "\nawait window.DaSiWaH3Forge.open(window.testNode); window.opened=true;</script></body>"
workspace = tempfile.TemporaryDirectory(prefix="dasiwa-forge-ui-")
fixture_path = Path(workspace.name) / "forge.html"
fixture_path.write_text(fixture)
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(fixture_path.as_uri())
    page.wait_for_function("window.opened === true")
    assert page.get_by_label("Picture 9 character", exact=True).input_value() == "character-9"
    shots = page.locator('select[title="How many shots. Auto lets the model choose."]')
    shots.select_option("3")
    assert page.get_by_label("Shot 3", exact=True).count() == 1
    page.get_by_label("Shot 1", exact=True).fill("Character 1 waves")
    page.get_by_label("Shot 3", exact=True).fill("Character 9 leaves")
    shots.select_option("1")
    assert page.get_by_label("Shot 3", exact=True).count() == 0
    shots.select_option("3")
    assert page.get_by_label("Shot 3", exact=True).input_value() == "Character 9 leaves"
    page.evaluate("window.DaSiWaH3Forge.close(window.testNode)")
    page.evaluate("window.DaSiWaH3Forge.open(window.testNode)")
    shots.select_option("3")
    assert page.get_by_label("Shot 1", exact=True).input_value() == "Character 1 waves"
    assert page.get_by_label("Shot 3", exact=True).input_value() == "Character 9 leaves"
    page.get_by_role("button", name="Generate", exact=True).click()
    page.wait_for_function("document.querySelector('.ds-forge .status')?.textContent.startsWith('Done')")
    assert page.get_by_role("button", name="Apply to node", exact=True).is_enabled()
    saved = page.evaluate("JSON.parse(JSON.stringify(window.testNode.properties))")
    page.add_init_script("window.restoredProperties = " + json.dumps(saved))
    page.reload()
    page.wait_for_function("window.opened === true")
    assert shots.input_value() == "3"
    assert page.get_by_label("Shot 3", exact=True).input_value() == "Character 9 leaves"
    assert page.get_by_role("button", name="Apply to node", exact=True).is_enabled()
    page.evaluate("window.DaSiWaH3Forge.close(window.testNode); const draft=window.testNode.properties.dasiwaH3ForgeHistory[0]; delete draft.draftOptions.shot_briefs; delete draft.forgeInputKey; window.DaSiWaH3Forge.open(window.testNode)")
    page.wait_for_function("document.querySelector('.ds-forge pre')?.textContent === 'Fixture draft'")
    assert page.get_by_label("Shot 3", exact=True).input_value() == ""
    page.evaluate("window.DaSiWaH3Forge.close(window.testNode); window.testNode={...window.testNode,properties:{}}; window.DaSiWaH3Forge.open(window.testNode)")
    page.wait_for_function("document.querySelector('.ds-forge .status')?.textContent.startsWith('Ready')")
    shots.select_option("3")
    assert page.get_by_label("Shot 3", exact=True).input_value() == ""
    page.evaluate("window.DaSiWaH3Forge.close(window.testNode); window.testNode.__dasiwaH3Forge.continuity=()=>({clip_id:'fixture',use_references:false}); window.DaSiWaH3Forge.open(window.testNode)")
    page.wait_for_function("document.querySelector('.ds-forge .status')?.textContent.startsWith('Ready')")
    assert not shots.is_visible()
    assert page.get_by_label("Shot 1", exact=True).count() == 0
    assert not errors, errors
    print("PASS: Chromium DOM — nine distinct references; shot-count fields; count-change and close/reopen retention; draft save/reload and Apply; same-ID node isolation; continuity hides Shots; no JS errors")
    browser.close()
workspace.cleanup()
