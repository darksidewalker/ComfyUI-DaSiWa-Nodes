import { app } from "../../scripts/app.js";

// DaSiWa LLM Prompt Writer: start_with is one small fixed box, the idea box
// is about two and a half times it and takes any extra height the node gets.
const START_WITH_HEIGHT = 56;
// Widget heights include about 20 px of padding, so 56 and 110 draw text
// areas of roughly 36 and 90: the 2.5x asked for.
const IDEA_MIN_HEIGHT = 110;

app.registerExtension({
  name: "DaSiWa.LLMPromptWriter",
  nodeCreated(node) {
    if (node.comfyClass !== "DaSiWa_LLMPromptWriter") return;
    const widget = name => node.widgets?.find(w => w.name === name);
    const startWith = widget("start_with");
    const idea = widget("idea");
    if (!startWith?.options || !idea?.options) return;
    startWith.options.getMinHeight = () => START_WITH_HEIGHT;
    startWith.options.getMaxHeight = () => START_WITH_HEIGHT;
    idea.options.getMinHeight = () => IDEA_MIN_HEIGHT;
    // A new node opens at the size that fits; a loaded one keeps its own.
    requestAnimationFrame(() => {
      const [width, height] = node.computeSize();
      if (node.size[1] < height) node.setSize([Math.max(node.size[0], width), height]);
      node.setDirtyCanvas?.(true, true);
    });
  },
});
