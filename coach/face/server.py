"""Face UI transport (T1.9): HTTP server for the page + WebSocket for live state.

server -> page : {"type": "state"|"mouth"|"subtitle"|"gaze"|"hud"|"panel"|"nod", ...}
page -> server : {"type": "event", "name": "consent"|"program"|"done"|"stop", "value": ...}

The last message of each type is replayed to new connections, so a page refresh
(or the kiosk browser restarting) restores the current face immediately.
"""
from __future__ import annotations

import functools
import json
import queue
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

UI_DIR = Path(__file__).parent / "ui"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class FaceServer:
    def __init__(self, host: str = "127.0.0.1", ws_port: int = 8765):
        self.host, self.ws_port, self.http_port = host, ws_port, ws_port + 1
        self.events: queue.Queue[dict] = queue.Queue()
        self._clients: set = set()
        self._last: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ws = None
        self._http = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.http_port}/index.html?ws={self.ws_port}"

    def start(self) -> "FaceServer":
        self._ws = serve(self._handler, self.host, self.ws_port)
        threading.Thread(target=self._ws.serve_forever, name="face-ws", daemon=True).start()
        handler = functools.partial(_QuietHandler, directory=str(UI_DIR))
        self._http = ThreadingHTTPServer((self.host, self.http_port), handler)
        threading.Thread(target=self._http.serve_forever, name="face-http", daemon=True).start()
        return self

    def stop(self) -> None:
        if self._ws:
            self._ws.shutdown()
        if self._http:
            self._http.shutdown()
            self._http.server_close()

    def _handler(self, conn) -> None:
        with self._lock:
            self._clients.add(conn)
            replay = list(self._last.values())
        try:
            for msg in replay:
                conn.send(json.dumps(msg))
            for raw in conn:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if msg.get("type") == "event":
                    self.events.put(msg)
        except ConnectionClosed:
            pass
        finally:
            with self._lock:
                self._clients.discard(conn)

    def send(self, msg: dict) -> None:
        data = json.dumps(msg, ensure_ascii=False)
        with self._lock:
            if msg["type"] != "nod":
                self._last[msg["type"]] = msg
            clients = list(self._clients)
        for c in clients:
            try:
                c.send(data)
            except ConnectionClosed:
                pass

    @property
    def connected(self) -> int:
        with self._lock:
            return len(self._clients)

    def next_event(self, timeout: float | None = None) -> dict | None:
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_events(self) -> None:
        while not self.events.empty():
            self.events.get_nowait()
