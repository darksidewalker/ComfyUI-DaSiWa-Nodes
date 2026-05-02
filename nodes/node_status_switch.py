"""
DaSiWa Node Status Switch

Mutes or bypasses target nodes based on a boolean toggle.

Targets are identified by wiring any output from the node you want
to control into one of this node's target inputs.  Inputs appear
dynamically: only one empty slot is shown at a time.  When you
connect it, the next slot appears (up to 99).

The switch exposes a single BOOLEAN output (`enabled_out`) carrying
its effective `enabled` value so a chain of switches can be driven
from a single upstream toggle.  Each switch in a chain still applies
its own `trigger_on` and `action` independently.
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

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("enabled_out",)
    FUNCTION = "execute"
    CATEGORY = "DaSiWa/utils"
    OUTPUT_NODE = True

    def execute(self, enabled, trigger_on, action, unique_id=None, **kwargs):
        # Pass the effective enabled value through to any chained switch.
        # Downstream switches apply their own trigger_on / action logic.
        return (bool(enabled),)
