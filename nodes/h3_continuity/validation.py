"""Validate the actual queued capture path before spending time on inference."""


def validate_capture_graph(prompt, guide_id):
    if prompt is None or guide_id is None:  # Direct Python use, outside a Comfy queue.
        return
    nodes = {str(key): value for key, value in prompt.items()}

    def linked(value, node_id, slot):
        return isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]) == str(node_id) and value[1] == slot

    appends = [(key, node) for key, node in nodes.items()
               if node.get("class_type") == "DaSiWaH3ContinuityAppend"
               and linked(node.get("inputs", {}).get("context"), guide_id, 2)]
    if len(appends) != 1:
        raise ValueError("Wire this Director Guide's continuity_context to one Append & Stage node.")
    append_id, append = appends[0]
    sampled = append.get("inputs", {}).get("sampled")
    if not isinstance(sampled, (list, tuple)) or len(sampled) != 2:
        raise ValueError("Wire the H3 sampler output to Append & Stage.sampled.")
    sampler = nodes.get(str(sampled[0]), {})
    if sampler.get("class_type") == "SamplerCustomAdvanced" and not linked(sampler.get("inputs", {}).get("latent_image"), guide_id, 1):
        raise ValueError("Append & Stage must use the sampler driven by this Director Guide.")
    publishes = [node for node in nodes.values()
                 if node.get("class_type") == "DaSiWaH3ContinuityPublish"
                 and linked(node.get("inputs", {}).get("ticket"), append_id, 1)]
    if len(publishes) != 1:
        raise ValueError("Wire Append & Stage.ticket to one Publish Export node.")
    filename = publishes[0].get("inputs", {}).get("filename")
    if not isinstance(filename, (list, tuple)) or len(filename) != 2:
        raise ValueError("Publish Export.filename must be wired to the actual video exporter.")
    exporter_id = str(filename[0])
    exporter = nodes.get(exporter_id, {})

    def depends_on(node_id, ancestor, visited=None):
        visited = set() if visited is None else visited
        if node_id == ancestor:
            return True
        if node_id in visited:
            return False
        visited.add(node_id)
        return any(depends_on(str(value[0]), ancestor, visited)
                   for value in nodes.get(node_id, {}).get("inputs", {}).values()
                   if isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]) in nodes)

    if not depends_on(exporter_id, append_id):
        raise ValueError("The selected exporter does not receive this cumulative latent. Check the Publish filename wire.")
    if exporter.get("class_type") == "DaSiWa_EnhancedVideoCombine":
        images = exporter.get("inputs", {}).get("images")
        if not isinstance(images, (list, tuple)) or len(images) != 2 or not depends_on(str(images[0]), append_id):
            raise ValueError("The exporter's images must come from this cumulative latent, not just its audio.")
        if exporter.get("inputs", {}).get("container") in {"Animated WebP", "Animated AVIF"}:
            raise ValueError("Select a video container for audiovisual continuity export.")
        for key in ("pingpong", "crop_to_audio"):
            if exporter.get("inputs", {}).get(key, False) is not False:
                raise ValueError(f"Turn off {key} on the continuity video exporter so its ending matches the saved latent.")
