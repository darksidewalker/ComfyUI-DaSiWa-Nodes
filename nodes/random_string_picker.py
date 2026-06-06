import random
import re


_RANDOM_GROUP_RE = re.compile(r"\{([^{}]*)\}")


class DaSiWa_RandomStringPicker:
    DESCRIPTION = (
        "DaSiWa Random String Picker: passes text through while replacing each "
        "{A|B|C} group with one randomly selected option."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "forceInput": True,
                        "description": "Text input. Each {A|B|C} group is replaced with one random option.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "pick"
    CATEGORY = "DaSiWa/Text"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return random.random()

    def pick(self, text):
        if text is None:
            return ("",)

        def replace_group(match):
            options = match.group(1).split("|")
            return random.choice(options)

        return (_RANDOM_GROUP_RE.sub(replace_group, str(text)),)
