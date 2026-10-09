"""Local transport regressions for a prospective, non-frozen research phase."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from research.aaa_erudition_v1.transport import LoopbackRuntime, RuntimeUnavailable


class LoopbackRuntimeTests(unittest.TestCase):
    def test_misleading_or_nonliteral_loopback_url_is_refused(self) -> None:
        bad_urls = (
            "http://127.0.0.1:80@attacker.invalid:8080",
            "http://localhost:8080",
            "http://127.0.0.1:8080/path",
            "http://127.0.0.1:8080/?token=x",
            "http://127.0.0.1:8080#fragment",
            "http://127.0.0.1:8080:90",
            "https://127.0.0.1:8080",
            "http://127.0.0.2:8080",
        )
        for url in bad_urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                LoopbackRuntime(url, Path("/does/not/exist"))

    def test_redirect_does_not_forward_bearer_header(self) -> None:
        captured: list[str | None] = []

        class Capture(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                captured.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *_args: object) -> None:
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Capture) as target:
            target_thread = threading.Thread(target=target.serve_forever, daemon=True)
            target_thread.start()

            class Redirect(BaseHTTPRequestHandler):
                def do_GET(self) -> None:
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{target.server_port}/capture")
                    self.end_headers()

                def log_message(self, *_args: object) -> None:
                    pass

            with ThreadingHTTPServer(("127.0.0.1", 0), Redirect) as source:
                source_thread = threading.Thread(target=source.serve_forever, daemon=True)
                source_thread.start()
                with tempfile.TemporaryDirectory() as tmp:
                    key = Path(tmp) / "key"
                    key.write_text("test-secret")
                    client = LoopbackRuntime(f"http://127.0.0.1:{source.server_port}", key)
                    with self.assertRaises(RuntimeUnavailable):
                        client.runtime()
                source.shutdown()
                source_thread.join(timeout=5)
            target.shutdown()
            target_thread.join(timeout=5)
        self.assertEqual(captured, [])

    def test_local_json_request_sends_key_to_selected_listener(self) -> None:
        seen: list[tuple[str, str | None, dict[str, object]]] = []

        class Answer(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, self.headers.get("Authorization"), body))
                data = b'{"choices":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args: object) -> None:
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Answer) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            with tempfile.TemporaryDirectory() as tmp:
                key = Path(tmp) / "key"
                key.write_text("test-secret")
                result = LoopbackRuntime(f"http://127.0.0.1:{server.server_port}", key).chat(
                    {"prompt": "probe"}
                )
            server.shutdown()
            worker.join(timeout=5)
        self.assertEqual(seen, [("/v1/chat/completions", "Bearer test-secret", {"prompt": "probe"})])
        self.assertEqual(result["choices"], [])
        self.assertGreaterEqual(result["_elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
