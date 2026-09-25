# MiniMax H3 Forge

**Forge** writes your H3 prompt for you with a local AI model (an LLM). Type a short idea, pick a model, press **Generate**. Forge fills the Director's prompt fields, **unloads the model to free your graphics card**, and then you press Run as usual.

The **Forge** button is on the MiniMax H3 Director node, next to its other buttons. It is not available in Image Inpaint mode.

---

## Quick start (if you don't know what to pick, do this)

1. Install **[Ollama](https://ollama.com/download)** and start it.
2. Download a model that can see pictures. In a terminal:
   ```
   ollama pull qwen3-vl:8b
   ```
3. Open the Director, press **Forge**, and pick the model from the **Model** list. The list is re-read every time Forge opens, so there is no need to restart ComfyUI.

That's it. There's nothing to set in ComfyUI's Settings for this, because Forge finds Ollama on your own computer automatically.

**Why Ollama?** It unloads the model the moment Forge is done, so H3 gets your whole graphics card, and it can look at your reference pictures.

---

## The three ways to run the model

The **Model** list groups models by where they come from. You only need one.

| Where the model runs | Setup | Frees your graphics card when done | Can see your pictures |
|---|---|---|---|
| **Ollama** (recommended) | Install Ollama, `ollama pull` a model | Yes | Yes, with a vision model |
| **Inside ComfyUI** (`ComfyUI/models/llm`) | Put the model folder or `.gguf` file there | Yes | Model folders: yes, if it is a vision model. `.gguf`: no, text only |
| **OpenAI-compatible server** (llama.cpp, llama-swap, LM Studio, koboldcpp) | Start the server, put its address in Settings | **Only with llama-swap**; see below | Yes, with a vision model |

### Ollama

- Forge looks for Ollama at `http://127.0.0.1:11434` (Ollama's normal address on your PC).
- If Ollama runs on another computer, put that computer's address in **Settings → DaSiWa → H3 Forge → Ollama address**, e.g. `http://192.168.1.50:11434`.
- A model that can see pictures has "vision" in its Ollama details. Forge checks this for you and only sends pictures to models that can see them.

### Inside ComfyUI (`models/llm`)

- Put the model in `ComfyUI/models/llm/`, either a Hugging Face model folder or a `.gguf` file.
- `.gguf` files need `llama-cpp-python` installed in ComfyUI's Python. Without it, they show in the list greyed out, with "needs llama-cpp-python installed".
- `.gguf` files run **text only**: Forge writes from your idea but can't look at your reference pictures. Model folders of a vision model can see them.

### OpenAI-compatible server (llama.cpp and friends)

1. Start your server. Note its address, e.g. `http://127.0.0.1:8080`.
2. In ComfyUI open **Settings → DaSiWa → H3 Forge**, and put the address in the box called **OpenAI-compatible server address**. Not the Ollama box.
3. Close and reopen Forge. The server's models appear in the **Model** list.

> **Important for a single graphics card:** Forge can only unload a model from **llama-swap**. A plain `llama-server`, LM Studio or koboldcpp keeps the model in your graphics card's memory after Forge is done. Forge then shows *"WARNING: model still loaded"*, and H3 will probably run out of memory.
> **Fix:** close the server (or unload the model in its own window) before you press Run. Or use llama-swap, which Forge can unload, or Ollama.

If the server is on **another computer**, the model uses that computer's graphics card, so nothing needs freeing on yours and the warning doesn't appear. That computer's server must accept connections from your network. A server started with `127.0.0.1` only answers the computer it runs on.

---

## Something went wrong

| What you see | What it means | What to do |
|---|---|---|
| The **Model** list is empty | Forge found no model anywhere | Do the Quick start above, then close and reopen Forge |
| A red note: *"Could not reach … Check the address in Settings > DaSiWa > H3 Forge."* | That source couldn't be reached at the address in Settings | Check the program is running and the address is right. For another computer, check it accepts network connections |
| Your server's models don't appear | The address is in the wrong box, or the server isn't running | Use **OpenAI-compatible server address**, not **Ollama address** |
| **"WARNING: model still loaded"** after Generate | Your server can't unload the model | Close the server or unload the model yourself before Run (see above) |
| A model is greyed out: "needs llama-cpp-python installed" | A `.gguf` in `models/llm` without llama.cpp support | Install `llama-cpp-python` in ComfyUI's Python, or use Ollama |
| The prompt ignores your pictures | The model can't see images (or it's a `.gguf` inside ComfyUI) | Pick a vision model, e.g. `qwen3-vl:8b` in Ollama |

---

## What Forge sends

Your idea, the Director's mode, duration and settings, and, for models that can see, the pictures on the Director's timeline. Nothing is sent anywhere except the model source you picked. Addresses live only in ComfyUI's Settings, never in the workflow, so a shared workflow can't point your ComfyUI at someone else's server.
