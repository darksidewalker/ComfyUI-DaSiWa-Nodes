"""Prune older Registry versions without depending on the latest publish being indexed."""

import json
import os
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.comfy.org"
DELETED = "NodeVersionStatusDeleted"


def request(method, path, token=None, payload=None):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Registry {method} {path}: HTTP {error.code}") from error


def list_versions(node):
    versions = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"nodeId": node, "page": page, "pageSize": 100})
        result = request("GET", f"/versions?{query}")
        if not isinstance(result, dict) or not isinstance(result.get("versions"), list):
            raise RuntimeError("Unexpected registry version listing")
        versions.extend(result["versions"])
        if page >= result["totalPages"]:
            if len(versions) != result["total"]:
                raise RuntimeError("Incomplete registry version listing; refusing to prune")
            return versions
        page += 1


def prune(node, publisher, current, token):
    # The just-published version may not be indexed (or may still be flagged).
    # Never mutate it; only operate on older versions already in the listing.
    versions = [v for v in list_versions(node) if v["status"] != DELETED and v["version"] != current]
    versions.sort(key=lambda v: (v["createdAt"], v["version"]), reverse=True)
    if len({v["id"] for v in versions}) != len(versions):
        raise RuntimeError("Duplicate version IDs; refusing to prune")
    prefix = f"/publishers/{urllib.parse.quote(publisher, safe='')}/nodes/{urllib.parse.quote(node, safe='')}/versions/"
    for version in versions[:4]:
        if version["deprecated"]:
            continue
        path = prefix + urllib.parse.quote(version["id"], safe="")
        request("PUT", path, token, {"deprecated": True})
        print(f"Deprecated {version['version']}")
    for version in versions[4:]:
        path = prefix + urllib.parse.quote(version["id"], safe="")
        request("DELETE", path, token)
        print(f"Unpublished {version['version']}")
    remaining = [v for v in list_versions(node) if v["status"] != DELETED and v["version"] != current]
    expected = {v["id"] for v in versions[:4]}
    if {v["id"] for v in remaining} != expected or any(not v["deprecated"] for v in remaining):
        raise RuntimeError("Registry cleanup verification failed")
    print(f"Verified {len(remaining)} deprecated predecessors remain; excluded newly published {current}")


if __name__ == "__main__":
    with open("pyproject.toml", "rb") as file:
        config = tomllib.load(file)
    token = os.environ.get("COMFY_REGISTRY_TOKEN")
    if not token:
        sys.exit("COMFY_REGISTRY_TOKEN is required")
    prune(config["project"]["name"], config["tool"]["comfy"]["PublisherId"], config["project"]["version"], token)
