#!/usr/bin/env python3
"""Re-baseline tests/fixtures/h3_legacy_node_contract.json after an intentional ABI change.

The fixture is the saved-workflow ABI guard of the pre-segment H3 nodes
(.github/scripts/test_h3_segment_compatibility.py). When a node's inputs change on
purpose, run this from the pack root and commit the fixture together with the node:

    python .github/scripts/sync_h3_node_contract.py            # report drift, change nothing
    python .github/scripts/sync_h3_node_contract.py --write    # rewrite the fixture
"""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "h3_legacy_node_contract.json"
KEYS = {"RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY", "OUTPUT_NODE", "OUTPUT_IS_LIST"}


def canonical(node):
    """Field-wise form used by the guard test, so serialization is deterministic."""
    if isinstance(node, ast.AST):
        return {"_type": type(node).__name__, **{
            name: canonical(value) for name, value in ast.iter_fields(node)
            if value is not None and value != []
        }}
    if isinstance(node, list):
        return [canonical(value) for value in node]
    return node


def current(filename, class_names):
    """Same traversal as the guard test: INPUT_TYPES bodies plus the pinned class attributes."""
    contracts = {}
    for cls in ast.parse((ROOT / filename).read_text()).body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in class_names:
            continue
        contract = {}
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "INPUT_TYPES":
                contract[node.name] = canonical(node)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in KEYS:
                        contract[target.id] = canonical(node.value)
        contracts[cls.name] = contract
    return contracts


def text_nodes(contract):
    """Every string constant in a recorded contract: names, options, defaults and tooltips."""
    if isinstance(contract, dict):
        found = [contract["value"]] if contract.get("_type") == "Constant" and isinstance(contract.get("value"), str) else []
        return found + [value for child in contract.values() for value in text_nodes(child)]
    if isinstance(contract, list):
        return [value for child in contract for value in text_nodes(child)]
    return []


def report(filename, old, new, changed):
    """Name what appeared or vanished, so an accidental widget change is visible in review."""
    for name in changed:
        before, after = text_nodes(old[name]), text_nodes(new[name])
        print(f"  {filename}::{name}")
        print(f"    added:   {[x for x in after if x not in before] or 'none'}")
        print(f"    dropped: {[x for x in before if x not in after] or 'none'}")
        for key in sorted(KEYS):
            if json.dumps(old[name].get(key)) != json.dumps(new[name].get(key)):
                print(f"    {key} changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite the fixture with the current contracts")
    args = parser.parse_args()
    fixture = json.loads(FIXTURE.read_text())
    refreshed, total = {}, []
    for filename, class_names in fixture.items():
        contracts = current(filename, class_names)
        missing = sorted(set(class_names) - set(contracts))
        if missing:
            raise SystemExit(f"{filename}: class(es) {missing} no longer exist; remove them from the fixture deliberately.")
        refreshed[filename] = contracts
        changed = [name for name in class_names
                   if json.dumps(contracts[name], sort_keys=True) != json.dumps(fixture[filename][name], sort_keys=True)]
        if changed:
            report(filename, fixture[filename], contracts, changed)
        total += [f"{filename}::{name}" for name in changed]
    if not total:
        print("contract=unchanged")
        return 0
    print("contract=changed classes=" + ",".join(total))
    if not args.write:
        print("rerun with --write after reviewing the ABI change above")
        return 1
    # Compact single-line JSON, like the recorded fixture: it is machine-written by this script.
    FIXTURE.write_text(json.dumps(refreshed, separators=(",", ":"), sort_keys=True))
    print(f"wrote {FIXTURE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
