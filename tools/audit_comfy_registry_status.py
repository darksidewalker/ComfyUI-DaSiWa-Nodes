"""Print a redacted summary of the DaSiWa Comfy Registry version statuses."""

import json
from collections import Counter
from urllib.request import urlopen


NODE_ID = "ComfyUI-DaSiWa-Nodes"
API_URL = (
    "https://api.comfy.org/versions?nodeId=ComfyUI-DaSiWa-Nodes"
    "&include_status_reason=true&pageSize=100"
)


def _finding_counts(status_reason):
    if not isinstance(status_reason, str):
        return {}
    try:
        reason = json.loads(status_reason)
        if isinstance(reason, list):
            findings = reason
        else:
            history = reason.get("statusHistory", [])
            message = history[-1].get("message", "") if history else ""
            findings = json.loads(message)
    except (IndexError, json.JSONDecodeError, TypeError, AttributeError):
        return {}
    if not isinstance(findings, list):
        return {}
    return dict(sorted(Counter(
        finding.get("issue_type")
        for finding in findings
        if isinstance(finding, dict) and isinstance(finding.get("issue_type"), str)
    ).items()))


def _policy(status_reason):
    if not isinstance(status_reason, str):
        return ""
    try:
        message = str(json.loads(status_reason).get("message", ""))
        return message.split(" — ", 1)[0]
    except (json.JSONDecodeError, AttributeError):
        return ""


def summarize_versions(payload):
    """Return public policy messages and finding counts without scan detail."""
    versions = payload.get("versions") if isinstance(payload, dict) else None
    if not isinstance(versions, list):
        raise ValueError("Registry response has no versions list.")
    return {
        "node_id": NODE_ID,
        "versions": [
            {
                "version": str(version.get("version", "")),
                "status": str(version.get("status", "")),
                "policy": _policy(version.get("status_reason")),
                "finding_counts": _finding_counts(version.get("status_reason")),
            }
            for version in versions
            if isinstance(version, dict)
        ],
    }


def require_active_version(report, version):
    """Raise unless the exact published Registry version is active."""
    for item in report["versions"]:
        if item["version"] == version:
            if item["status"] == "NodeVersionStatusActive":
                return
            raise ValueError(
                f"Registry version {version} is {item['status']}, not active."
            )
    raise ValueError(f"Registry version {version} was not found.")


def main():
    try:
        with urlopen(API_URL, timeout=30) as response:
            payload = json.load(response)
        print(json.dumps(summarize_versions(payload), indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Registry status audit failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
