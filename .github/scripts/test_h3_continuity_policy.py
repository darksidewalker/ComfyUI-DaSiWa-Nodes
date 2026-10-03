"""Lightweight continuity-policy regression checks; no ComfyUI or torch required."""
import ast
import importlib
import json
from pathlib import Path
from typing import Any
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
# Load the real pure helpers without importing the ComfyUI node-pack entrypoint.
PACKAGE = "_dasiwa_policy_nodes"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "nodes")]
sys.modules[PACKAGE] = package
builder = importlib.import_module(f"{PACKAGE}.helper_minimax_h3_prompt_builder")
forge = importlib.import_module(f"{PACKAGE}.h3_forge")


def policy_namespace():
    source = ast.parse((ROOT / "nodes/h3_continuity/core.py").read_text())
    chosen = [node for node in source.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "DEFAULT_PROMPT" for target in node.targets
    ) or isinstance(node, ast.FunctionDef) and node.name in {
        "_window_context", "compose_ref_fields", "compose_prompt"}]
    namespace: dict[str, Any] = {"__name__": f"{PACKAGE}.h3_continuity.core",
                                 "__package__": f"{PACKAGE}.h3_continuity"}
    exec(compile(ast.Module(body=chosen, type_ignores=[]), "core.py", "exec"), namespace)
    return namespace


class ContinuityPolicyTests(unittest.TestCase):
    def setUp(self):
        self.compose = policy_namespace()["compose_prompt"]
        self.settings = {"version": 3, "overlap_frames": 22, "extension_frames": 119,
                         "continuation_prompt": "", "idea": ""}

    def test_auto_welds_seam_then_continues_naturally(self):
        prompt = self.compose(self.settings)
        self.assertIn("hidden overlapping context", prompt)
        self.assertIn("seam", prompt)
        self.assertIn("continue the established action, camera motion, and sound naturally", prompt.lower())
        self.assertNotIn("Next action:", prompt)

    def test_written_next_action_can_change_pace_and_camera_after_seam(self):
        prompt = self.compose({**self.settings, "continuation_prompt": "She sprints; camera pans to follow."})
        self.assertIn("Next action: She sprints; camera pans to follow.", prompt)
        self.assertIn("after the overlap", prompt.lower())
        self.assertIn("camera", prompt.lower())
        self.assertNotIn("Do not invent a pan", prompt)

    def test_legacy_v2_custom_prompt_remains_authoritative(self):
        prompt = self.compose({**self.settings, "version": 2, "continuation_prompt": "A deliberate dolly shot."})
        self.assertIn("A deliberate dolly shot.", prompt)
        self.assertNotIn("Continue the same uninterrupted shot naturally.", prompt)

    def test_idea_without_written_prompt_is_not_treated_as_auto(self):
        prompt = self.compose({**self.settings, "idea": "Accelerate after the seam."})
        self.assertIn("Next action: Accelerate after the seam.", prompt)
        self.assertNotIn("With no next action", prompt)

    def test_structured_prompt_preserves_definitions_and_empty_music(self):
        fields = {"subject_definitions": "<Subject 1> is the rider.",
                  "summary": "The rider turns.", "retention_analysis": "Keep the same coat.",
                  "detailed_description": "The rider turns toward the camera.",
                  "soundscape": "Hooves and wind.", "music": ""}
        authored = builder.build_ref_prompt({"ref": fields}, preserve_empty=True)
        settings = {**self.settings, "continuation_prompt": authored}
        parsed = builder.parse_ref_prompt(self.compose(settings))
        self.assertEqual(parsed["subject_definitions"], fields["subject_definitions"])
        self.assertEqual(parsed["retention_analysis"], fields["retention_analysis"])
        self.assertEqual(parsed["music"], "")
        self.assertEqual(parsed["detailed_description"].count("This is a continuation window."), 1)
        self.assertIn(fields["detailed_description"], parsed["detailed_description"])
        self.assertEqual(settings["continuation_prompt"], authored)

    def test_legacy_structured_migration_keeps_idea_out_of_music(self):
        script = """import {pathToFileURL} from 'node:url';
const {normalizeContinuity} = await import(pathToFileURL(process.argv[1]).href);
let input = ''; for await (const chunk of process.stdin) input += chunk;
const c = JSON.parse(input), duration = {value: 5};
normalizeContinuity(c, duration);
const first = JSON.stringify(c);
normalizeContinuity(c, duration);
if (JSON.stringify(c) !== first) throw Error('Migration is not idempotent');
console.log(JSON.stringify(c));
"""
        for music, uppercase, aliases in (("", False, False), ("Quiet strings.", False, False),
                                           ("", True, False), ("Quiet strings.", True, True)):
            with self.subTest(music=music, uppercase=uppercase, aliases=aliases):
                fields = {"subject_definitions": "<Subject 1> is the rider.",
                          "summary": "The rider turns.", "retention_analysis": "Keep the coat.",
                          "detailed_description": "The rider turns toward the camera.",
                          "soundscape": "Hooves and wind.", "music": music}
                authored = builder.build_ref_prompt({"ref": fields}, preserve_empty=True)
                if aliases:
                    authored = authored.replace("overall_soundscape:", "soundscape:").replace("non_diegetic_music:", "music:")
                if uppercase:
                    authored = "\n".join(line.upper() if line.endswith(":") else line for line in authored.split("\n"))
                raw = {"version": 2, "operation": "continue", "source_id": "parent",
                       "extension_frames": 119, "continuation_prompt": authored,
                       "idea": "Speed up after the seam."}
                result = subprocess.run(
                    ["node", "--input-type=module", "-e", script,
                     str(ROOT / "js/minimax_h3_continuity.js")],
                    input=json.dumps(raw), text=True, capture_output=True, check=True)
                migrated = json.loads(result.stdout)
                self.assertEqual(migrated["version"], 3)
                parsed = builder.parse_ref_prompt(migrated["continuation_prompt"])
                self.assertIsNotNone(parsed)
                for key in fields.keys() - {"detailed_description"}:
                    self.assertEqual(parsed[key], fields[key])
                self.assertIn("Speed up after the seam.", parsed["detailed_description"])
                composed = builder.parse_ref_prompt(self.compose({**self.settings, **migrated}))
                self.assertEqual(composed["music"], music)
                self.assertEqual(composed["detailed_description"].count("Speed up after the seam."), 1)

    def test_v3_normalization_preserves_auto_and_custom_prompts(self):
        script = """import {pathToFileURL} from 'node:url';
const m = await import(pathToFileURL(process.argv[1]).href);
if ('CONTINUITY_LEGACY_PROMPT' in m) process.exit(1);
const auto = m.normalizeContinuity({version:3, continuation_prompt:'', source_id:'', operation:'continue'});
if (auto.continuation_prompt !== '' || auto.operation !== 'new') process.exit(2);
const custom = m.normalizeContinuity({version:3, source_id:'source', continuation_prompt:'Pan after the seam.', operation:'new', extension_frames:119});
if (custom.continuation_prompt !== 'Pan after the seam.' || custom.operation !== 'continue' || 'extension_frames' in custom) process.exit(3);
"""
        subprocess.run(["node", "--input-type=module", "-e", script, str(ROOT / "js/minimax_h3_continuity.js")], check=True)

    def test_forge_allows_requested_changes_without_restarting(self):
        source = ast.parse((ROOT / "nodes/h3_forge.py").read_text())
        assignment = next(node for node in source.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "CONTINUATION_SYSTEM" for target in node.targets))
        system = ast.literal_eval(assignment.value).lower()
        self.assertIn("overlap", system)
        self.assertIn("after the seam", system)
        self.assertIn("camera", system)
        self.assertIn("next action", system)
    def test_forge_without_new_idea_respects_existing_next_action(self):
        backend = Mock(base="http://remote.invalid")
        backend.models.return_value = [{"id": "ollama:test"}]
        backend.can_see.return_value = False
        backend.chat.return_value = ("She sprints while the camera pans.", {})
        backend.unload.return_value = True
        with patch.object(forge, "backends", return_value={"ollama": backend}), \
             patch.object(forge, "_is_this_machine", return_value=False):
            forge.generate_continuity_draft({"prompt": "old clip"}, "", "/unused", "ollama:test", {},
                                            current_prompt="She sprints; camera pans after the seam.")
        user = backend.chat.call_args.args[2]
        self.assertIn("She sprints; camera pans after the seam.", user)
        self.assertIn("Follow the current next-action draft", user)
        self.assertNotIn("New idea: Continue the current action naturally.", user)


if __name__ == "__main__":
    unittest.main()
