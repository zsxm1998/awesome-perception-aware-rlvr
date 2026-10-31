# Copyright 2026 the Awesome-Perception-Aware-RLVR authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Serve an agent environment over HTTP with the Python standard library (docs/agent_environments.md).

An environment server writes one ``Episode`` class; ``EpisodeServer`` does what every server needs: episode
ids, ``step_index`` deduplication (a retried step gets its first response back instead of running twice),
idempotent ``/finalize`` and ``/close``, and dropping episodes idle for longer than ``idle_timeout_s`` (a
crashed trainer never closes them). It is a reference implementation for the examples and tests; a production
server can use any web framework that speaks the same JSON.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional, Protocol

from PIL import Image


class Episode(Protocol):
    """One episode of an environment. Return values are the JSON bodies of docs/agent_environments.md."""

    def reset(self, config: dict[str, Any], seed: int, max_tool_calls: int) -> dict[str, Any]:
        """{"user_content": [...], "tool_schemas": [...], "system_prompt": str | None, "state_key"?: str}"""

    def step(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """{"content": [...], "success": bool, "error_code"?, "done"?, "reward"?, "state_key"?, "cost"?, "info"?}"""

    def finalize(self, final_answer: Optional[str], status: str) -> dict[str, Any]:
        """{"reward": float | None, "components": {...}, "metrics": {...}, "details"?: {...}}"""


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(image: Image.Image) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"type": "image", "png_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}


class ProtocolError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class _Record:
    episode: Any
    touched: float
    steps: dict[int, dict[str, Any]] = field(default_factory=dict)
    outcome: Optional[dict[str, Any]] = None
    # An episode method raised: its effects are unknown, so the episode takes no further calls and a repeated
    # request is answered with this failure instead of running the method again.
    failure: Optional[str] = None
    busy: int = 0  # requests in progress; an episode in use is never dropped as idle
    lock: threading.Lock = field(default_factory=threading.Lock)


class EpisodeServer:
    """Idle episodes are dropped when a later request arrives (no request, no cleanup)."""

    def __init__(
        self,
        make_episode: Callable[[], Episode],
        *,
        idle_timeout_s: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.make_episode = make_episode
        self.idle_timeout_s = idle_timeout_s
        self.clock = clock
        self._episodes: dict[str, _Record] = {}
        self._lock = threading.Lock()

    def handle(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self._drop_idle_episodes()
        if path == "/reset":
            episode = self.make_episode()
            response = episode.reset(dict(body["config"]), int(body["seed"]), int(body.get("max_tool_calls", 0)))
            episode_id = uuid.uuid4().hex
            with self._lock:
                self._episodes[episode_id] = _Record(episode=episode, touched=self.clock())
            return {"episode_id": episode_id, **response}
        if path not in ("/step", "/finalize", "/close"):
            raise ProtocolError(HTTPStatus.NOT_FOUND, f"unknown endpoint {path}")
        with self._lock:
            record = self._episodes.get(body.get("episode_id"))
            if record is not None:
                record.busy += 1
        if record is None:
            if path == "/close":
                return {}
            raise ProtocolError(HTTPStatus.NOT_FOUND, f"unknown or expired episode {body.get('episode_id')!r}")
        try:
            with record.lock:
                return self._handle_episode(path, body, record)
        finally:
            with self._lock:
                record.busy -= 1
                record.touched = self.clock()

    def _handle_episode(self, path: str, body: dict[str, Any], record: _Record) -> dict[str, Any]:
        if path == "/close":
            with self._lock:
                self._episodes.pop(body["episode_id"], None)
            return {}
        if path == "/step":
            index = body["step_index"]
            if index in record.steps:
                return record.steps[index]
            self._refuse_failed(record)
            if index != len(record.steps):
                raise ProtocolError(
                    HTTPStatus.CONFLICT, f"step_index {index} out of order (next is {len(record.steps)})"
                )
            call = body["tool_call"]
            try:
                result = record.episode.step(call["name"], call["arguments"])
            except Exception as exc:
                record.failure = f"step {index} raised {type(exc).__name__}: {exc}"
                raise
            record.steps[index] = {"done": False, "cost": 0.0, "info": {}, **result}
            return record.steps[index]
        if record.outcome is None:
            self._refuse_failed(record)
            try:
                outcome = record.episode.finalize(body["final_answer"], body["status"])
            except Exception as exc:
                record.failure = f"finalize raised {type(exc).__name__}: {exc}"
                raise
            record.outcome = {"components": {}, "metrics": {}, **outcome}
        return record.outcome

    @staticmethod
    def _refuse_failed(record: _Record) -> None:
        if record.failure is not None:
            raise ProtocolError(HTTPStatus.CONFLICT, f"the episode failed and takes no more calls: {record.failure}")

    def _drop_idle_episodes(self) -> None:
        deadline = self.clock() - self.idle_timeout_s
        with self._lock:
            for episode_id in [
                key for key, record in self._episodes.items() if record.busy == 0 and record.touched < deadline
            ]:
                del self._episodes[episode_id]

    def http_server(self, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep connections alive between requests

            def do_POST(self) -> None:  # noqa: N802 (http.server naming)
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    status, payload = HTTPStatus.OK, server.handle(self.path, body)
                except ProtocolError as exc:
                    status, payload = exc.status, {"error": str(exc)}
                except (KeyError, TypeError, ValueError) as exc:
                    status, payload = HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"}
                except Exception as exc:  # an environment error: 500, which the client retries
                    status, payload = HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"}
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:  # requests are not logged; errors are returned
                return

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 1024  # a rollout batch opens many connections at once

        return Server((host, port), Handler)


def serve(make_episode: Callable[[], Episode], *, host: str, port: int, idle_timeout_s: float = 1800.0) -> None:
    http_server = EpisodeServer(make_episode, idle_timeout_s=idle_timeout_s).http_server(host, port)
    print(f"[agent_env] serving on http://{host}:{http_server.server_address[1]}", flush=True)
    http_server.serve_forever()
