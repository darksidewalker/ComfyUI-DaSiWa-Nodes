"""Technical post-sampler continuation nodes; Director owns all controls."""
import logging
from pathlib import Path
from .core import ClipStore, append_tail
log = logging.getLogger(__name__)

class DaSiWaH3ContinuityAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampled": ("LATENT",), "context": ("DF_H3_CONTINUITY_CONTEXT",)}}

    RETURN_TYPES = ("LATENT", "DF_H3_CONTINUITY_TICKET")
    RETURN_NAMES = ("cumulative_latent", "ticket")
    FUNCTION = "commit"
    CATEGORY = "DaSiWa/MiniMax H3"

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
    CATEGORY = "DaSiWa/MiniMax H3"

    def publish(self, filename, ticket):
        if ticket.get("disabled"):
            return ("Continuity capture is off.",)
        import folder_paths
        path = Path(filename).resolve()
        roots = [Path(folder_paths.get_output_directory()).resolve(), Path(folder_paths.get_temp_directory()).resolve()]
        if not any(path.is_relative_to(root) for root in roots) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError("The video exporter did not produce a valid output file; continuity was not advanced.")
        store = ClipStore()
        # Also catch an incorrect playback rate or a downstream time trim. FPS
        # interpolation is fine when the exported duration remains unchanged.
        import av
        metadata = store.inspect(ticket["session"], ticket["clip_id"], ready=False)
        with av.open(str(path)) as media:
            video = next(iter(media.streams.video), None)
            if video is None:
                raise ValueError("Continuity export has no video stream.")
            seconds = (float(video.duration * video.time_base) if video.duration is not None
                       else float(media.duration or 0) / av.time_base)
            if abs(seconds - metadata["seconds"]) > max(0.125, 2 / float(video.average_rate or 24)):
                raise ValueError("Export duration differs from the saved H3 timeline. Check FPS, interpolation and trimming; checkpoint remains staged.")
        data = store.publish(ticket, path)
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
