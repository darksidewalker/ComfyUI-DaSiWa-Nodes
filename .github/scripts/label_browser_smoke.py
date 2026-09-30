"""Run against an isolated ComfyUI: python .github/scripts/label_browser_smoke.py http://127.0.0.1:8199.
Test-only dependencies: playwright and Pillow. No runtime dependencies.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen

from PIL import Image
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8199"
TYPE = "Lable (DaSiWa)"
source = Path(__file__).resolve().parents[2] / "js" / "dasiwa_label.js"
assert urlopen(URL + os.environ.get("LABEL_TEST_ASSET", "/extensions/ComfyUI-DaSiWa-Nodes/dasiwa_label.js")).read() == source.read_bytes()

with tempfile.TemporaryDirectory(prefix="label-smoke-", dir=os.environ.get("TMPDIR")) as temp, sync_playwright() as p:
    options = {"headless": True, "args": ["--no-sandbox"]}
    if os.environ.get("LABEL_TEST_CHROME"):
        options["executable_path"] = os.environ["LABEL_TEST_CHROME"]
    browser = p.chromium.launch(**options)
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, locale="de-DE")
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(URL, wait_until="networkidle")
    page.wait_for_function("!!window.LiteGraph?.registered_node_types['Lable (DaSiWa)']", timeout=60000)
    page.evaluate("async () => { window.testApp = (await import('/scripts/app.js')).app; }")
    assert page.evaluate("() => { const before=LiteGraph.registered_node_types['Lable (DaSiWa)']; const extension=testApp.extensions.find(e=>e.name==='DaSiWa.Lable'); extension.registerCustomNodes(); return before===LiteGraph.registered_node_types['Lable (DaSiWa)'] && document.querySelectorAll('#dasiwa-label-style').length===1; }")
    for vue in [False, True]:
        page.evaluate("vue => testApp.ui.settings.setSettingValue('Comfy.VueNodes.Enabled', vue)", vue)
        page.evaluate("() => { testApp.graph.clear(); const n=LiteGraph.createNode('Lable (DaSiWa)'); n.title='Test label'; n.pos=[100,100]; testApp.graph.add(n); window.labelNode=n; }")
        root = page.locator(".dasiwa-label")
        root.wait_for(state="visible")
        page.wait_for_timeout(300)
        bounds = root.bounding_box()
        assert bounds["width"] > 250 and bounds["height"] > 100, bounds
        if vue:
            assert page.locator(".lg-node:has(.dasiwa-label)").count() == 1
            assert page.locator(".lg-node:has(.dasiwa-label) .lg-node-header").count() == 0
        before_drag = page.evaluate('Array.from(labelNode.pos)')
        page.mouse.move(bounds['x'] + 25, bounds['y'] + 50)
        page.mouse.down()
        page.mouse.move(bounds['x'] + 125, bounds['y'] + 110, steps=8)
        page.mouse.up()
        after_drag = page.evaluate('Array.from(labelNode.pos)')
        assert after_drag[0] - before_drag[0] > 50 and after_drag[1] - before_drag[1] > 30, ('drag failed', vue, before_drag, after_drag, errors)
        page.evaluate('() => { labelNode.pos=[100,100]; labelNode.setDirtyCanvas(true,true); }')
        page.wait_for_timeout(200)
        color_result = page.evaluate("() => { const n=labelNode; n.properties.backgroundOpacity=0; for (const option of Object.values(LGraphCanvas.node_colors)) { n.setColorOption(option); if(n.color!=='transparent'||n.bgcolor!=='transparent'||n.properties.backgroundOpacity!==0) throw Error('Native color overrode transparency'); } n.properties.backgroundOpacity=0.4; n.setColorOption(LGraphCanvas.node_colors.red); const visible=getComputedStyle(n.layer).backgroundColor; const selected=n.properties.backgroundColor; n.setColorOption(null); return {visible,selected,cleared:n.properties.backgroundOpacity,wrapper:n.bgcolor}; }")
        assert color_result['visible'].endswith(', 0.4)')
        assert len(color_result['selected']) == 7
        assert color_result['cleared'] == 0 and color_result['wrapper'] == 'transparent'
        print('Native node colors preserve label opacity: PASS')
        root.click(button='right', position={'x':30,'y':30})
        page.get_by_text('Edit label…', exact=True).click()
        page.locator('.dasiwa-label-editor').wait_for(state='visible')
        page.get_by_role('button', name='Done', exact=True).click()
        page.locator('.dasiwa-label-editor').wait_for(state='detached')
        root.dblclick()
        editor = page.locator(".dasiwa-label-editor")
        editor.wait_for(state="visible")
        menu_size = editor.evaluate("el => ({width:el.getBoundingClientRect().width,scrollWidth:el.scrollWidth,clientWidth:el.clientWidth,scrollHeight:el.scrollHeight,clientHeight:el.clientHeight})")
        print('Options menu:', menu_size)
        assert menu_size['width'] == 680
        assert menu_size['scrollWidth'] <= menu_size['clientWidth']
        assert menu_size['scrollHeight'] <= menu_size['clientHeight']
        page.set_viewport_size({'width':640,'height':600})
        compact = editor.evaluate("el => ({width:el.getBoundingClientRect().width,top:el.getBoundingClientRect().top,bottom:el.getBoundingClientRect().bottom,scrollWidth:el.scrollWidth,clientWidth:el.clientWidth,scrollHeight:el.scrollHeight,clientHeight:el.clientHeight})")
        assert compact['width'] <= 640 * 0.9 + 1
        assert compact['top'] >= 0 and compact['bottom'] <= 600
        assert compact['scrollWidth'] <= compact['clientWidth']
        assert compact['scrollHeight'] > compact['clientHeight']
        page.set_viewport_size({'width':1440,'height':1000})
        assert editor.locator('h2').inner_text() == TYPE
        assert editor.get_attribute('lang') == 'en'
        assert editor.locator('.image-picker span').inner_text() == 'No image selected'
        assert editor.locator('input[type=file]').is_hidden()
        assert editor.locator('.eyedropper').count() == 2
        assert editor.locator('.eyedropper svg').count() == 2
        assert editor.locator('.eyedropper').evaluate_all("els => els.every(el => el.textContent === '' && el.getAttribute('aria-label').startsWith('Pick '))")
        font = page.locator('[data-setting="fontFamily"]')
        assert font.evaluate("el => [...el.options].every(option => option.style.fontFamily === option.value || option.style.fontFamily === '\"'+option.value+'\"')")
        page.get_by_label("Label text", exact=True).fill("Header\\nLong text wraps around embedded picture. " * 8)
        page.locator('[data-setting="fontFamily"]').select_option("Georgia")
        assert font.evaluate("el => getComputedStyle(el).fontFamily") == 'Georgia'
        font.select_option('Courier New')
        assert 'Courier New' in font.evaluate("el => getComputedStyle(el).fontFamily")
        font.select_option('Georgia')
        page.locator('[data-setting="textAlign"]').select_option("right")
        assert editor.locator('.swatch').count() == 96
        page.get_by_role('button', name='Background #e6e6fa', exact=True).click()
        assert page.evaluate('labelNode.properties.backgroundColor') == '#e6e6fa'
        page.get_by_role("button", name="Text color #22c55e", exact=True).click()
        assert page.get_by_label('Text color HEX', exact=True).input_value() == '#22c55e'
        for label, key in [('Text color', 'fontColor'), ('Background', 'backgroundColor')]:
            hex_field = page.get_by_label(label+' HEX', exact=True)
            for entry, expected in [('#A1B2C3', '#a1b2c3'), ('#f80', '#ff8800'), ('123abc', '#123abc')]:
                hex_field.fill(entry)
                hex_field.press('Enter')
                assert hex_field.input_value() == expected
                assert page.evaluate(f'labelNode.properties.{key}') == expected
                assert page.locator(f'[data-setting="{key}"]').input_value() == expected
            hex_field.fill('invalid')
            hex_field.press('Enter')
            assert hex_field.input_value() == '#123abc'
            assert page.evaluate(f'labelNode.properties.{key}') == '#123abc'
            assert editor.locator('.status').inner_text() == 'Enter a HEX color such as #ff8800 or #f80.'
            hex_field.fill('#22c55e')
            hex_field.press('Tab')
            assert page.evaluate(f'labelNode.properties.{key}') == '#22c55e'
            page.locator(f'[data-setting="{key}"]').evaluate("el => { el.value='#112233'; el.dispatchEvent(new Event('input')); }")
            assert hex_field.input_value() == '#112233'
        for setting, value in [('fontSize',24), ('padding',16), ('borderRadius',12), ('fontOpacity',0.9), ('imageSize',45), ('imageOpacity',0.8), ('width',400), ('height',240)]:
            page.locator(f'[data-setting="{setting}"]').evaluate("(el,v) => { el.value=String(v); el.dispatchEvent(new Event('input')); }", value)
        numbers = editor.locator('.slider input[type=number]')
        assert numbers.count() == editor.locator('input[type=range]').count()
        for label, setting, value in [('Font size','fontSize',31), ('Width','width',455), ('Height','height',275), ('Image opacity','imageOpacity',0.33), ('Rotation','angle',-25)]:
            field = page.get_by_label(label+' value', exact=True)
            field.fill(str(value))
            field.press('Enter')
            assert float(field.input_value()) == value
            assert float(page.locator(f'[data-setting="{setting}"]').input_value()) == value
        assert page.evaluate('labelNode.properties.fontSize') == 31
        assert page.evaluate('Array.from(labelNode.size)') == [455,275]
        # Range edits must update the number field too.
        page.locator('[data-setting="fontSize"]').evaluate("el => { el.value='24'; el.dispatchEvent(new Event('input')); }")
        assert page.get_by_label('Font size value',exact=True).input_value() == '24'
        page.get_by_label('Font size value',exact=True).fill('')
        page.get_by_label('Font size value',exact=True).press('Enter')
        assert page.get_by_label('Font size value',exact=True).input_value() == '24'
        state = page.evaluate("() => ({...labelNode.properties, size:Array.from(labelNode.size)})")
        assert state['size'] == [455,275]
        assert state['padding'] == 16 and state['borderRadius'] == 12 and state['fontOpacity'] == 0.9
        page.locator('[data-setting="imageFit"]').select_option("cover")
        assert page.evaluate("labelNode.properties.imageFit") == "cover"
        page.locator('[data-setting="imageFit"]').select_option("contain")
        page.locator('[data-setting="backgroundOpacity"]').evaluate("el => { el.value='0.75'; el.dispatchEvent(new Event('input')); }")
        page.locator('[data-setting="angle"]').evaluate("el => { el.value='15'; el.dispatchEvent(new Event('input')); }")
        assert page.locator('.dasiwa-label-text').evaluate("el=>getComputedStyle(el).fontFamily") == "Georgia"
        assert "\n" in page.locator('.dasiwa-label-text').inner_text()
        assert page.locator('.dasiwa-label-layer').evaluate("el=>el.style.transform") == "rotate(15deg)"
        # Exercise native EyeDropper integration; real monitor sampling needs user permission.
        page.evaluate("() => { window.EyeDropper=class { async open(){ return {sRGBHex:'#123456'}; } }; }")
        page.get_by_role("button", name="Pick text color from screen", exact=True).evaluate("el=>el.disabled=false")
        page.get_by_role("button", name="Pick text color from screen", exact=True).click()
        page.wait_for_function("labelNode.properties.fontColor==='#123456'")
        assert page.get_by_label('Text color HEX', exact=True).input_value() == '#123456'
        page.locator('[data-setting="angle"]').evaluate("el => { el.value='0'; el.dispatchEvent(new Event('input')); }")
        for extension, fmt in [("png", "PNG"), ("jpg", "JPEG"), ("webp", "WEBP")]:
            image_path = Path(temp) / ("picture." + extension)
            Image.new("RGB", (800, 400), "#f97316").save(image_path, fmt)
            with page.expect_file_chooser() as picker:
                page.get_by_role('button', name='Choose image…', exact=True).click()
            picker.value.set_files(image_path)
            mime = 'jpeg' if extension == 'jpg' else extension
            page.wait_for_function("mime => labelNode.properties.image.startsWith('data:image/'+mime+';base64,') && labelNode.picture.complete && labelNode.picture.naturalWidth === 800", arg=mime)
            assert "embedded" in editor.locator('.status').inner_text()
            for mode in ["background", "float left", "float right", "above text", "below text"]:
                page.locator('[data-setting="imageMode"]').select_option(mode)
                geometry = page.evaluate("() => { const a=labelNode.picture.getBoundingClientRect(), b=labelNode.root.getBoundingClientRect(); return {width:a.width,height:a.height,rootWidth:b.width,rootHeight:b.height,float:labelNode.picture.style.float}; }")
                assert geometry['width'] <= geometry['rootWidth'] * 1.3
                assert geometry['height'] <= geometry['rootHeight'] * 1.3
                if mode.startswith("float"):
                    assert geometry['float'] == mode.split()[1]
        embedded = page.evaluate("labelNode.properties.image")
        editor.locator('input[type="file"]').set_input_files({'name':'bad.svg','mimeType':'image/svg+xml','buffer':b'<svg/>'})
        page.wait_for_function("document.querySelector('.dasiwa-label-editor .status').textContent === 'Choose PNG, JPEG or WebP.'")
        assert page.evaluate("labelNode.properties.image") == embedded
        page.locator('[data-setting="angle"]').evaluate("el => { el.value='0'; el.dispatchEvent(new Event('input')); }")
        page.locator('[data-setting="imageMode"]').select_option("float left")
        page.locator('[data-setting="textAlign"]').select_option("left")
        page.get_by_role("button", name="Done", exact=True).click()
        editor.wait_for(state='detached')
        for target in [root, page.locator('.dasiwa-label-text'), page.locator('.dasiwa-label-image')]:
            target.hover(position={'x':5,'y':5}, force=True)
            assert target.evaluate("el => getComputedStyle(el).cursor") == 'crosshair'
        workflow = page.evaluate("() => testApp.graph.serialize()")
        original = workflow["nodes"][0]
        assert original["properties"]["image"].startswith("data:image/webp;base64,")
        workflow['nodes'][0]['color'] = '#ff0000'
        workflow['nodes'][0]['bgcolor'] = '#ff0000'
        page.evaluate("async data => { await testApp.loadGraphData(data); window.labelNode=testApp.graph._nodes.find(n=>n.type==='Lable (DaSiWa)'); }", workflow)
        page.locator('.dasiwa-label').wait_for(state="visible")
        page.wait_for_function("labelNode.picture.complete && labelNode.picture.naturalWidth === 800")
        restored = page.evaluate("() => labelNode.serialize()")
        assert restored['color'] == 'transparent' and restored['bgcolor'] == 'transparent'
        assert restored["properties"] == original["properties"]
        assert restored["title"] == original["title"]
        assert restored["size"] == original["size"], (original["size"], restored["size"])
        prompt = page.evaluate("async () => (await testApp.graphToPrompt()).output")
        assert not any(node["class_type"] == TYPE for node in prompt.values())
        page.evaluate("() => { labelNode.setSize([520,320]); labelNode.render(); }")
        page.wait_for_timeout(200)
        assert page.locator('.dasiwa-label').bounding_box()['width'] > bounds['width']
        # Pin must pass left clicks to the node underneath, while right click still opens settings.
        page.evaluate("() => { const n=LiteGraph.createNode('Note'); n.pos=[100,100]; n.size=[200,120]; testApp.graph.add(n); testApp.graph.remove(labelNode); testApp.graph.add(labelNode); labelNode.pin(true); }")
        page.wait_for_timeout(200)
        page.evaluate("() => { window.underlyingHits=0; const n=testApp.graph._nodes.find(n=>n.type==='Note'); const el=document.querySelector('.lg-node[data-node-id=\"'+n.id+'\"]'); if(el) el.addEventListener('pointerdown',()=>window.underlyingHits++); }")
        rect = page.locator('.dasiwa-label').bounding_box()
        page.mouse.click(rect['x'] + 20, rect['y'] + 20)
        if vue:
            assert page.evaluate("underlyingHits") > 0
        assert page.evaluate("() => labelNode.isPointInside(110,110)") is False
        page.locator('.dasiwa-label').click(button="right", position={"x": 300, "y": 200})
        page.locator('.dasiwa-label-editor').wait_for(state="visible")
        page.get_by_label("Pin label", exact=True).uncheck()
        page.get_by_role("button", name="Done", exact=True).click()
        page.evaluate("() => testApp.graph.remove(testApp.graph._nodes.find(n=>n.type==='Note'))")
        page.screenshot(path=str(Path(temp) / ("vue.png" if vue else "classic.png")))
        # Untrusted workflows cannot load remote URLs or markup.
        page.evaluate("() => { labelNode.properties.image='https://example.com/image.png'; labelNode.title='<img src=x onerror=alert(1)>'; labelNode.render(); }")
        assert page.locator('.dasiwa-label-image').get_attribute('src') is None
        assert page.locator('.dasiwa-label-text img').count() == 0
        page.evaluate("() => labelNode.edit()")
        page.get_by_label('Label text', exact=True).fill('Short\\nTwo')
        page.get_by_role('button', name='Fit to text', exact=True).click()
        fit = page.evaluate("() => ({size:Array.from(labelNode.size), p:labelNode.properties})")
        assert abs(fit['size'][1] - (2 * fit['p']['fontSize'] * 1.2 + 2 * fit['p']['padding'])) < 0.1
        page.get_by_role('button', name='Remove image', exact=True).click()
        assert page.evaluate('labelNode.properties.image') == ''
        page.keyboard.press('Escape')
        page.locator('.dasiwa-label-editor').wait_for(state='detached')
        assert page.locator('.dasiwa-label-editor').count() == 0
        page.evaluate("() => { labelNode.edit(); testApp.graph.remove(labelNode); }")
        page.locator('.dasiwa-label-editor').wait_for(state='detached')
        assert page.locator('.dasiwa-label-editor').count() == 0
        page.locator('.dasiwa-label').wait_for(state='detached')
        print(('Nodes 2.0' if vue else 'Classic') + ': drag, numeric fields, font previews, palette, images, layouts, reload, resize, pin, prompt exclusion, validation, cleanup PASS')
    assert not errors, json.dumps(errors)
    browser.close()
print('PASS: served asset matches source; no uncaught browser errors')
