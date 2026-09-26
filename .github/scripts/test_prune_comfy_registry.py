"""Offline checks for Registry pruning while a fresh publish is not indexed."""

import unittest
from unittest.mock import patch

import prune_comfy_registry as registry


class PruneTests(unittest.TestCase):
    def check_prune(self, current_is_listed):
        versions = [
            {"id": str(i), "version": f"0.4.{i}", "createdAt": f"2026-09-{i:02d}T00:00:00Z",
             "status": "NodeVersionStatusFlagged", "deprecated": False}
            for i in range(1, 7)
        ]
        if current_is_listed:
            versions.append({"id": "7", "version": "0.4.7", "createdAt": "2026-09-07T00:00:00Z",
                             "status": "NodeVersionStatusPending", "deprecated": False})
        mutations = []

        def request(method, path, token=None, payload=None):
            version = next(v for v in versions if path.endswith("/" + v["id"]))
            mutations.append((method, version["id"]))
            if method == "PUT":
                assert payload is not None
                version["deprecated"] = payload["deprecated"]
            else:
                version["status"] = registry.DELETED

        with patch.object(registry, "list_versions", side_effect=lambda node: [v.copy() for v in versions]), \
             patch.object(registry, "request", side_effect=request):
            registry.prune("node", "publisher", "0.4.7", "token")
        self.assertEqual(mutations, [("PUT", "6"), ("PUT", "5"), ("PUT", "4"), ("PUT", "3"),
                                     ("DELETE", "2"), ("DELETE", "1")])
        self.assertTrue(all(v["deprecated"] for v in versions if v["id"] in {"3", "4", "5", "6"}))
        if current_is_listed:
            self.assertFalse(versions[-1]["deprecated"])

    def test_current_missing_from_index(self):
        self.check_prune(False)

    def test_current_pending_in_index(self):
        self.check_prune(True)


if __name__ == "__main__":
    unittest.main()
