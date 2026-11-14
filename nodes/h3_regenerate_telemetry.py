"""Execution-scoped H3 phase diagnostics; telemetry never changes the processing result."""
from contextlib import contextmanager
import logging
import time
import uuid

_LOG = logging.getLogger(__name__)
_EVENT = 'dasiwa_h3_regenerate_stage'


def make_telemetry():
    context = None
    try:
        from comfy_execution.utils import get_executing_context
        current = get_executing_context()
        if (current is not None and str(uuid.UUID(current.prompt_id)) == current.prompt_id
                and isinstance(current.node_id, str) and 0 < len(current.node_id) <= 128
                and all(c.isascii() and (c.isalnum() or c in '_-:.') for c in current.node_id)):
            context = {'prompt_id': current.prompt_id, 'node': current.node_id}
    except (ImportError, AttributeError, ValueError, TypeError):
        pass
    warned = False

    def emit(data):
        nonlocal warned
        severity = logging.ERROR if data['outcome'] == 'failed' else logging.INFO
        # Fields come only from phase() call sites, never prompts, model names or exception text.
        _LOG.log(severity, 'H3 Hi-Res Regenerate prompt_id=%s node=%s %s',
                 context['prompt_id'] if context else 'unavailable',
                 context['node'] if context else 'unavailable',
                 ' '.join(f'{key}={value}' for key, value in data.items()))
        if context is None:
            return
        try:
            from server import PromptServer
            server = PromptServer.instance
            # Never turn missing ownership into a broadcast to other clients.
            if server is not None and server.client_id:
                server.send_sync(_EVENT, {**data, **context}, server.client_id)
        except Exception:
            if not warned:
                warned = True
                _LOG.warning('H3 Hi-Res Regenerate telemetry transport unavailable; local phase logs retained')
    return emit


@contextmanager
def phase(telemetry, name, *, pass_index=1, chunk=1, chunks=1, **fields):
    if telemetry is None:
        yield
        return
    started = time.monotonic()

    def report(outcome):
        try:
            telemetry({'phase': name, 'outcome': outcome,
                       'pass': pass_index, 'chunk': chunk, 'chunks': chunks,
                       'elapsed_ms': max(0, int((time.monotonic() - started) * 1000)), **fields})
        except Exception:
            # Diagnostics must not mask successful processing or the original failure.
            pass

    report('started')
    try:
        yield
    except BaseException as error:
        from comfy.model_management import InterruptProcessingException
        report('cancelled' if isinstance(error, InterruptProcessingException) else 'failed')
        raise
    else:
        report('completed')
