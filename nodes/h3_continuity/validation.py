"""Validate the actual queued capture path before spending time on inference."""


def validate_capture_graph(prompt, guide_id, output_mode="full"):
    if prompt is None or guide_id is None:  # Direct Python use, outside a Comfy queue.
        return
    nodes = {str(key): value for key, value in prompt.items()}

    def linked(value, node_id, slot):
        return isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]) == str(node_id) and value[1] == slot

    segment = output_mode == "segment"
    # Segment Prepare also runs full timelines, so either writer is valid in full mode.
    writers = {"DaSiWaH3ContinuitySegmentPrepare"} if segment else {"DaSiWaH3ContinuityAppend", "DaSiWaH3ContinuitySegmentPrepare"}
    appends = [(key, node) for key, node in nodes.items()
               if node.get("class_type") in writers
               and linked(node.get("inputs", {}).get("context"), guide_id, 2)]
    if len(appends) != 1:
        raise ValueError("Wire this Director Guide's continuity_context to one Segment Prepare → Segment Decode → Export → Publish chain." if segment else "Wire this Director Guide's continuity_context to one Append & Stage or Segment Prepare node.")
    append_id, append = appends[0]
    ticket_slot = 2 if append.get("class_type") == "DaSiWaH3ContinuitySegmentPrepare" else 1
    sampled = append.get("inputs", {}).get("sampled")
    if not isinstance(sampled, (list, tuple)) or len(sampled) != 2:
        raise ValueError("Wire the H3 sampler output to Append & Stage.sampled.")
    def guide_video_latent(value, visited=None):
        """Follow only known video-carrying latent edges, never mask/conditioning edges."""
        if linked(value, guide_id, 1):
            return True
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        node_id, slot = str(value[0]), value[1]
        visited = set() if visited is None else visited
        edge = (node_id, slot)
        if edge in visited:
            return False
        visited.add(edge)
        node = nodes.get(node_id, {})
        inputs = node.get("inputs", {})
        latent_input = {
            "SetLatentNoiseMask": "samples",
            "LTXVSeparateAVLatent": "av_latent",
            "LTXVConcatAVLatent": "video_latent",
        }.get(node.get("class_type"))
        if latent_input is not None and slot == 0:
            return guide_video_latent(inputs.get(latent_input), visited)
        return False

    sampler = nodes.get(str(sampled[0]), {})
    if sampler.get("class_type") == "SamplerCustomAdvanced" and not guide_video_latent(sampler.get("inputs", {}).get("latent_image")):
        raise ValueError("Append & Stage must use the sampler driven by this Director Guide.")
    publishes = [node for node in nodes.values()
                 if node.get("class_type") == "DaSiWaH3ContinuityPublish"
                 and linked(node.get("inputs", {}).get("ticket"), append_id, ticket_slot)]
    if len(publishes) != 1:
        raise ValueError("Wire Append & Stage.ticket to one Publish Export node.")
    filename = publishes[0].get("inputs", {}).get("filename")
    if not isinstance(filename, (list, tuple)) or len(filename) != 2:
        raise ValueError("Publish Export.filename must be wired to the actual video exporter.")
    exporter_id = str(filename[0])

    def media_exporter(node_id, seen):
        """Step back through filename pass-throughs (e.g. VHS_SelectFilename) to the node that took the images."""
        node = nodes.get(node_id, {})
        if node_id in seen or "images" in node.get("inputs", {}):
            return node_id
        seen.add(node_id)
        sources = [str(v[0]) for v in node.get("inputs", {}).values() if isinstance(v, (list, tuple)) and len(v) == 2 and str(v[0]) in nodes]
        return media_exporter(sources[0], seen) if len(sources) == 1 else node_id

    exporter_id = media_exporter(exporter_id, set())
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

    if segment:
        def render_latent(value, seen=None):
            if linked(value, append_id, 0):
                return True
            if not isinstance(value, (list, tuple)) or len(value) != 2 or value[1] != 0:
                return False
            key = str(value[0])
            seen = set() if seen is None else seen
            if key in seen:
                return False
            seen.add(key)
            node = nodes.get(key, {})
            inputs = node.get("inputs", {})
            return (node.get("class_type") == "DaSiWaH3TiledUpscale"
                    and linked(inputs.get("continuity_context"), append_id, 1)
                    and render_latent(inputs.get("latent"), seen))
        decoders = [(key,node) for key,node in nodes.items()
                    if node.get("class_type") == "DaSiWaH3ContinuitySegmentDecode"
                    and linked(node.get("inputs",{}).get("context"),append_id,1)
                    and render_latent(node.get("inputs",{}).get("latent"))]
        if len(decoders) != 1:
            raise ValueError("Wire Segment Prepare.render_latent and segment_context through optional Upscale to one Segment Decode.")
        decode_id = decoders[0][0]
        images = exporter.get("inputs",{}).get("images")
        # Spatial post-processing may sit between decode and exporter; Publish still checks 24 fps and frame count.
        if not (isinstance(images,(list,tuple)) and len(images) == 2 and depends_on(str(images[0]),decode_id)) or not linked(exporter.get("inputs",{}).get("audio"),decode_id,1):
            raise ValueError("Export images derived from, and audio directly from, the same Segment Decode.")
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
