"""Compare real Python/JavaScript timing policies without importing torch/ComfyUI."""
import ast
import json
import math
from pathlib import Path
import subprocess
from typing import Any
import unittest

ROOT = Path(__file__).resolve().parents[2]
source = ast.parse((ROOT / "nodes/h3_continuity/core.py").read_text())
function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "continuation_timing")
namespace: dict[str, Any] = {"math": math}
exec(compile(ast.Module(body=[function], type_ignores=[]), "core.py", "exec"), namespace)
timing = namespace["continuation_timing"]


class ContinuityTimingTests(unittest.TestCase):
    def test_frontend_backend_parity_at_boundaries_and_short_sources(self):
        cases = [[seconds, overlap, frames]
                 for seconds in (17 / 24, 119 / 24, 357 / 24, 15)
                 for overlap in (5, 22, 39, 56, 73)
                 for frames in (5, 21, 22, 38, 39, 55, 56, 72, 73, 124)]
        script = """import {pathToFileURL} from 'node:url';
const {continuityTiming} = await import(pathToFileURL(process.argv[1]).href);
let input=''; for await (const chunk of process.stdin) input+=chunk;
console.log(JSON.stringify(JSON.parse(input).map(args => continuityTiming(...args))));
"""
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script, str(ROOT / "js/minimax_h3_continuity.js")],
            input=json.dumps(cases), text=True, capture_output=True, check=True)
        actual = json.loads(result.stdout)
        self.assertEqual(actual, [timing(*case) for case in cases])
        for case, value in zip(cases, actual):
            self.assertLessEqual(value["window_frames"], 362, case)
            self.assertLessEqual(value["overlap_frames"], case[2], case)
            self.assertEqual(value["extension_frames"] % 17, 0, case)

    def test_all_valid_legacy_extensions_are_preserved(self):
        for frames in range(17, 358, 17):
            with self.subTest(frames=frames):
                self.assertEqual(timing(frames / 24)["extension_frames"], frames)

    def test_invalid_duration_and_too_short_sources_fail(self):
        for seconds in (0, -1, 15.01, math.nan, math.inf):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                timing(seconds)
        with self.assertRaisesRegex(ValueError, "at least 5"):
            timing(5, 22, 4)


if __name__ == "__main__":
    unittest.main()
