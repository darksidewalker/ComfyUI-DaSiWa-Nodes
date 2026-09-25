"""Technical post-sampler continuation nodes; Director owns all controls."""
import logging
from pathlib import Path
from .core import ClipStore, append_tail
from .media import make_tail_thumbnails
log = logging.getLogger(__name__)

class DaSiWaH3ContinuityAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampled": ("LATENT",), "context": ("DF_H3_CONTINUITY_CONTEXT",)}}

    RETURN_TYPES = ("LATENT", "DF_H3_CONTINUITY_TICKET")
    RETURN_NAMES = ("cumulative_latent", "ticket")
    FUNCTION = "commit"
    CATEGORY = "DF/MiniMax H3 Continuity"

    def commit(self, sampled, context):
        if context.get("disabled"):
            return sampled, {"disabled": True}
        if context["operation"] == "continue":
            previous, _ = ClipStore().load(context["session"], context["source_id"])
            combined = append_tail(previous, sampled, context["layout"])
        else:
            combined = sampled
        ticket = ClipStore().stage(combined, context)
        return combined, ticket


class DaSiWaH3ContinuityPublish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filename": ("STRING", {"forceInput": True}),
                             "ticket": ("DF_H3_CONTINUITY_TICKET",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "publish"
    OUTPUT_NODE = True
    CATEGORY = "DF/MiniMax H3 Continuity"

    def publish(self, filename, ticket):
        if ticket.get("disabled"):
            return ("Continuity capture is off.",)
        import folder_paths
        path = Path(filename).resolve()
        roots = [Path(folder_paths.get_output_directory()).resolve(), Path(folder_paths.get_temp_directory()).resolve()]
        if not any(path.is_relative_to(root) for root in roots) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError("The video exporter did not produce a valid output file; continuity was not advanced.")
        store = ClipStore()
        warning = ""
        try:
            thumbnails = make_tail_thumbnails(path, store.clip_dir(ticket["session"], ticket["clip_id"]))
        except Exception as exc:
            # Prompt assistance is optional; an unavailable preview must not invalidate
            # a successfully generated latent and exported movie.
            thumbnails = []
            warning = str(exc)
            log.warning("H3 continuity preview unavailable: %s", exc)
        data = store.publish(ticket, path, thumbnails, warning)
        message = f"Saved {data['frames']} frames ({data['seconds']:.3f}s), clip {data['clip_id'][:8]}."
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("df_h3_continuity_saved", {
                "session": ticket["session"], "clip_id": ticket["clip_id"], "status": message})
        except (ImportError, AttributeError):
            pass
        return {"ui": {"text": [message]}, "result": (message,)}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (DaSiWaH3ContinuityAppend, DaSiWaH3ContinuityPublish)}
NODE_DISPLAY_NAME_MAPPINGS = {"DaSiWaH3ContinuityAppend": "H3 Continuity • Append & Stage", "DaSiWaH3ContinuityPublish": "H3 Continuity • Publish Export"}
