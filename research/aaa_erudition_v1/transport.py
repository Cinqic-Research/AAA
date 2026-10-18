"""Bearer-authenticated transport restricted to a literal loopback listener.

This is a prospective successor to the frozen ``aaa.erudition.v0`` transport.
It is not used to reinterpret or replay v0 evidence.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class RuntimeUnavailable(RuntimeError):
    """The local model runtime failed or attempted to leave the trusted endpoint."""


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        _req: urllib.request.Request,
        _fp: Any,
        _code: int,
        _msg: str,
        _headers: Any,
        _newurl: str,
    ) -> None:
        raise RuntimeUnavailable("model runtime redirect refused")


def _loopback_base(url: str) -> str:
    if not url.isascii():
        raise ValueError("runtime URL must be ASCII")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("invalid runtime URL") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or port is None
        or port < 1
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("runtime must be an HTTP listener at a literal loopback address and explicit port")
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"http://{host}:{port}"


class LoopbackRuntime:
    """Send requests directly to a local model server without proxies or redirects."""

    def __init__(self, url: str, key_path: Path, timeout: float = 900.0) -> None:
        self.url = _loopback_base(url)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.timeout = timeout
        self.key = key_path.read_text("utf-8").strip()
        if not self.key:
            raise ValueError("empty runtime key")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RejectRedirect())

    def _json(self, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.key}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"{self.url}{path}", data=data, headers=headers)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as error:
            raise RuntimeUnavailable("model runtime request failed") from error
        if not isinstance(result, dict):
            raise RuntimeUnavailable("model runtime returned a non-object JSON response")
        return result

    def runtime(self) -> dict[str, Any]:
        props = self._json("/props", None, min(self.timeout, 60.0))
        keep = ("build_info", "model_path", "total_slots", "chat_template_caps")
        out = {key: props[key] for key in keep if key in props}
        settings = props.get("default_generation_settings", {})
        out["n_ctx"] = settings.get("n_ctx") if isinstance(settings, dict) else None
        return out

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        result = self._json("/v1/chat/completions", request, self.timeout)
        result["_elapsed_seconds"] = time.monotonic() - started
        return result
