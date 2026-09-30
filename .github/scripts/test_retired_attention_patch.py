"""Keep the retired attention patch out of the distributable nodepack."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RetiredAttentionPatchTest(unittest.TestCase):
    def test_patch_is_removed(self):
        source = (ROOT / "__init__.py").read_text()
        self.assertNotIn("PathchComfyKitchenAttentionDaSiWa", source)
        self.assertNotIn("nodes_comfy_kitchen_attention", source)
        self.assertFalse((ROOT / "nodes/nodes_comfy_kitchen_attention.py").exists())
        form = (ROOT / ".github/ISSUE_TEMPLATE/bug-report.yml").read_text()
        self.assertNotIn("Patch Comfy Kitchen Attention", form)


if __name__ == "__main__":
    unittest.main()
