"""Operator trust boundary and nonblocking server discovery regressions."""
import threading
from types import SimpleNamespace

import pytest
from comfy.cli_args import args
from nodes import llm_backends as backend


@pytest.fixture(autouse=True)
def synthetic_settings(monkeypatch):
    for name in ("DASIWA_LLM_ALLOW_SETTINGS", "DASIWA_LLM_OLLAMA_URL",
                 "DASIWA_LLM_OPENAI_URL", "DASIWA_LLM_OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(args, "multi_user", False)
    monkeypatch.setattr(backend, "comfy_settings", lambda: {
        "ollama_url": "http://user-ollama", "openai_url": "http://user-openai",
        "openai_api_key": "user-key"})
    if hasattr(backend, "_model_choices_cache"):
        backend._model_choices_cache.clear()
        monkeypatch.setattr(backend, "_model_choices_thread", None)
    yield
    join_discovery()


def join_discovery():
    thread = getattr(backend, "_model_choices_thread", None)
    if thread is not None:
        thread.join(2)
        assert not thread.is_alive()


def test_settings_not_trusted_by_default(monkeypatch):
    reads = []
    monkeypatch.setattr(backend, "comfy_settings", lambda: reads.append(True) or {})
    assert backend.workflow_server_settings() == {
        "ollama_url": backend.DEFAULT_OLLAMA, "openai_url": "", "openai_api_key": ""}
    assert reads == []


@pytest.mark.parametrize("opt_in", [None, "0", "true", "yes", " 1"])
def test_only_exact_operator_opt_in_enables_settings(monkeypatch, opt_in):
    if opt_in is not None:
        monkeypatch.setenv("DASIWA_LLM_ALLOW_SETTINGS", opt_in)
    assert backend.workflow_server_settings()["openai_url"] == ""


def test_single_user_operator_can_opt_in(monkeypatch):
    monkeypatch.setenv("DASIWA_LLM_ALLOW_SETTINGS", "1")
    assert backend.workflow_server_settings() == {
        "ollama_url": "http://user-ollama", "openai_url": "http://user-openai",
        "openai_api_key": "user-key"}


def test_multi_user_disables_settings_fallback(monkeypatch):
    monkeypatch.setenv("DASIWA_LLM_ALLOW_SETTINGS", "1")
    monkeypatch.setattr(args, "multi_user", True)
    reads = []
    monkeypatch.setattr(backend, "comfy_settings", lambda: reads.append(True) or {
        "openai_url": "http://user-openai", "openai_api_key": "user-key"})
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://operator/v1/")
    monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", " operator-key ")
    assert backend.workflow_server_settings() == {
        "ollama_url": backend.DEFAULT_OLLAMA, "openai_url": "http://operator/v1",
        "openai_api_key": "operator-key"}
    assert reads == []


def test_environment_url_never_uses_settings_key(monkeypatch):
    monkeypatch.setenv("DASIWA_LLM_ALLOW_SETTINGS", "1")
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://operator")
    assert backend.workflow_server_settings()["openai_api_key"] == ""


def test_settings_url_never_uses_environment_key(monkeypatch):
    monkeypatch.setenv("DASIWA_LLM_ALLOW_SETTINGS", "1")
    monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", "operator-key")
    assert backend.workflow_server_settings()["openai_api_key"] == "user-key"


def test_key_without_endpoint_is_not_retained(monkeypatch):
    monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", "operator-key")
    assert backend.workflow_server_settings()["openai_api_key"] == ""


def test_missing_workflow_endpoint_explains_operator_opt_in():
    with pytest.raises(ValueError, match="DASIWA_LLM_ALLOW_SETTINGS=1"):
        backend.run_workflow_server(
            {"backend": "openai", "model_path": "served"}, "s", "u", [],
            10, 0.2, 0.8, 1.0, -1)


def test_explicit_forge_backends_never_inherit_environment_key(monkeypatch):
    monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", "operator-key")
    assert backend.backends({"openai_url": "http://user"})["openai"].headers == {}
    assert backend.backends({"openai_url": "http://user", "openai_api_key": "user-key"})[
        "openai"].headers == {"Authorization": "Bearer user-key"}


def test_schema_discovery_does_not_call_network_on_caller(monkeypatch):
    caller = threading.current_thread()
    calls = []
    def http(*a, **kw):
        calls.append(threading.current_thread())
        return {"models": [{"name": "served"}]}
    monkeypatch.setattr(backend, "_http", http)
    assert backend.server_model_choices() == []
    thread = getattr(backend, "_model_choices_thread", None)
    if thread is not None:
        thread.join(2)
        assert not thread.is_alive()
    assert calls and all(t is not caller and t.daemon for t in calls)


@pytest.fixture
def discovery(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(backend, "time", SimpleNamespace(monotonic=lambda: clock[0]), raising=False)
    calls = []
    response = {"models": [{"name": "ollama-one"}], "data": [{"id": "server-one"}]}
    fail = set()
    def http(url, payload=None, timeout=10, headers=None):
        calls.append((url, headers, timeout))
        if any(marker in url for marker in fail):
            raise OSError("transient synthetic-secret")
        return response
    monkeypatch.setattr(backend, "_http", http)
    monkeypatch.setenv("DASIWA_LLM_OLLAMA_URL", "http://ollama")
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://openai")
    monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", "key-one")
    return clock, calls, response, fail


def test_explicit_refresh_forces_discovery_and_populates_schema_cache(discovery):
    refresh = getattr(backend, "refresh_server_model_choices", None)
    assert callable(refresh), "Forge needs an explicit off-thread refresh helper"
    _, calls, response, _ = discovery
    assert refresh() == ["Ollama: ollama-one", "Server: server-one"]
    assert backend.server_model_choices() == ["Ollama: ollama-one", "Server: server-one"]
    assert len(calls) == 2
    response["data"] = [{"id": "server-two"}]
    assert refresh() == ["Ollama: ollama-one", "Server: server-two"]
    assert len(calls) == 4
    assert calls[1] == ("http://openai/v1/models", {"Authorization": "Bearer key-one"}, backend.LIST_TIMEOUT)


def test_ttl_and_retry_preserve_each_last_good_backend(discovery):
    clock, calls, response, fail = discovery
    assert backend.server_model_choices() == []
    join_discovery()
    expected = ["Ollama: ollama-one", "Server: server-one"]
    assert backend.server_model_choices() == expected
    assert len(calls) == 2
    clock[0] += backend.MODEL_CHOICES_TTL + 1
    fail.add("openai")
    response["models"] = [{"name": "ollama-two"}]
    assert backend.server_model_choices() == expected
    join_discovery()
    expected = ["Ollama: ollama-two", "Server: server-one"]
    assert backend.server_model_choices() == expected
    assert len(calls) == 4
    clock[0] += backend.MODEL_CHOICES_RETRY + 1
    fail.clear()
    response["data"] = [{"id": "recovered"}]
    assert backend.server_model_choices() == expected
    join_discovery()
    assert backend.server_model_choices() == ["Ollama: ollama-two", "Server: recovered"]
    assert len(calls) == 5  # Only the failed backend was due for retry.


@pytest.mark.parametrize("change", ["url", "key"])
def test_changed_configuration_cannot_reuse_other_server_models(discovery, monkeypatch, change):
    assert backend.server_model_choices() == []
    join_discovery()
    _, calls, _, fail = discovery
    fail.add("openai")
    if change == "url":
        monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://openai-changed")
    else:
        monkeypatch.setenv("DASIWA_LLM_OPENAI_API_KEY", "key-two")
    assert backend.server_model_choices() == ["Ollama: ollama-one"]
    join_discovery()
    assert backend.server_model_choices() == ["Ollama: ollama-one"]
    assert len(calls) == 3
    assert calls[-1][1] == {"Authorization": "Bearer " + ("key-two" if change == "key" else "key-one")}


def test_single_worker_does_not_block_schema_calls_during_configuration_churn(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def http(url, **kwargs):
        calls.append(url)
        entered.set()
        assert release.wait(2), "test must release discovery"
        return {"models": [{"name": "old"}]}
    monkeypatch.setattr(backend, "_http", http)
    assert backend.server_model_choices() == []
    try:
        assert entered.wait(2)
        worker = backend._model_choices_thread
        assert worker.daemon
        for i in range(20):
            monkeypatch.setenv("DASIWA_LLM_OLLAMA_URL", f"http://changed-{i}")
            assert backend.server_model_choices() == []
            assert backend._model_choices_thread is worker
        assert len(calls) == 1
    finally:
        release.set()
        join_discovery()
    assert backend.server_model_choices() == []
    join_discovery()
    assert backend.server_model_choices() == ["Ollama: old"]
    assert calls == [backend.DEFAULT_OLLAMA + "/api/tags", "http://changed-19/api/tags"]


def test_invalid_config_neither_discovers_nor_leaks_previous_choices(discovery, monkeypatch):
    assert backend.server_model_choices() == []
    join_discovery()
    _, calls, _, _ = discovery
    monkeypatch.setenv("DASIWA_LLM_OPENAI_URL", "http://user:synthetic-secret@host")
    assert backend.server_model_choices() == []
    assert len(calls) == 2


def test_schema_returns_copy_of_cache(discovery):
    assert backend.server_model_choices() == []
    join_discovery()
    choices = backend.server_model_choices()
    choices.clear()
    assert backend.server_model_choices() == ["Ollama: ollama-one", "Server: server-one"]


def test_explicit_refresh_serializes_with_worker_without_blocking_schema(monkeypatch):
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    class ObservedCondition(threading.Condition):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)
    monkeypatch.setattr(backend, "_model_choices_condition", ObservedCondition())
    calls = []
    def http(url, **kwargs):
        calls.append(threading.current_thread())
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        return {"models": [{"name": "fresh"}]}
    monkeypatch.setattr(backend, "_http", http)
    results = []
    assert backend.server_model_choices() == []
    explicit = threading.Thread(target=lambda: results.append(backend.refresh_server_model_choices()))
    try:
        assert entered.wait(2)
        explicit.start()
        assert waiting.wait(2)
        assert len(calls) == 1
        assert backend.server_model_choices() == []
    finally:
        release.set()
        join_discovery()
        if explicit.ident is not None:
            explicit.join(2)
            assert not explicit.is_alive()
    assert results == [["Ollama: fresh"]]
    assert len(calls) == 2 and calls[0] is not calls[1]


def test_explicit_refresh_contention_has_bounded_wait(discovery, monkeypatch):
    _, calls, _, _ = discovery
    assert backend.refresh_server_model_choices() == ["Ollama: ollama-one", "Server: server-one"]
    waits = []
    def busy(predicate, timeout=None):
        waits.append(timeout)
        return False
    monkeypatch.setattr(backend._model_choices_condition, "wait_for", busy)
    assert backend.refresh_server_model_choices() == ["Ollama: ollama-one", "Server: server-one"]
    assert waits == [2 * backend.LIST_TIMEOUT + 1]
    assert len(calls) == 2


def test_successfully_empty_server_list_replaces_stale_models(discovery):
    _, _, response, _ = discovery
    backend.refresh_server_model_choices()
    response["data"] = []
    assert backend.refresh_server_model_choices() == ["Ollama: ollama-one"]


def test_start_failure_does_not_prevent_later_discovery(discovery, monkeypatch):
    _, calls, _, _ = discovery
    with monkeypatch.context() as patcher:
        def fail_start(self):
            raise RuntimeError("cannot create thread")
        patcher.setattr(threading.Thread, "start", fail_start)
        assert backend.server_model_choices() == []
    assert not calls
    assert backend.server_model_choices() == []
    join_discovery()
    assert backend.server_model_choices() == ["Ollama: ollama-one", "Server: server-one"]


def test_thread_constructor_failure_does_not_poison_discovery(discovery, monkeypatch):
    with monkeypatch.context() as patcher:
        def fail_constructor(**kwargs):
            raise RuntimeError("cannot construct thread")
        patcher.setattr(backend, "threading", SimpleNamespace(Thread=fail_constructor))
        assert backend.server_model_choices() == []
    assert backend.server_model_choices() == []
    join_discovery()
    assert backend.server_model_choices() == ["Ollama: ollama-one", "Server: server-one"]


def test_discovery_failures_never_log_credentials(discovery, monkeypatch, capsys):
    _, _, _, fail = discovery
    fail.update(("ollama", "openai"))
    logs = []
    monkeypatch.setattr(backend, "log_dasiwa", lambda *args: logs.append(args))
    assert backend.refresh_server_model_choices() == []
    assert logs == []
    captured = capsys.readouterr()
    assert "synthetic-secret" not in captured.out + captured.err
    assert "key-one" not in captured.out + captured.err
