"""
DaSiWa Node Status Switch

Mutes or bypasses target nodes based on a boolean toggle.

Targets are identified by wiring any output from the node you want
to control into one of this node's target inputs.  Inputs appear
dynamically: only one empty slot is shown at a time.  When you
connect it, the next slot appears (up to 20).

No outputs.  Pure control node.
"""


class AnyType(str):
    """Matches any ComfyUI type so target inputs accept anything."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __hash__(self):
        return hash("*")


ANY = AnyType("*")


class DaSiWa_NodeStatusSwitch:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "trigger_on": (
                    ["true \u2192 active", "false \u2192 active"],
                    {"default": "true \u2192 active"},
                ),
                "action": (["mute", "bypass"], {"default": "bypass"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "DaSiWa/utils"
    OUTPUT_NODE = True

    def execute(self, enabled, trigger_on, action, unique_id=None, **kwargs):
        return {}
