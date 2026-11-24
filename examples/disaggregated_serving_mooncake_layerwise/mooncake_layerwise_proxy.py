# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Layerwise Mooncake proxy: start Decode as soon as Prefill begins.

Engine id and bootstrap address are prefetched from the prefiller /query, so
Decode can be forwarded with the Prefill request. layerwise_notify_url remains
the fallback when a prefiller restart changes the engine id. Do not use this
proxy for bulk transfers.
"""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import httpx

_PENDING: dict[str, dict] = {}
_PENDING_LOCK = threading.Lock()
_PREFILL_URL = ""
_PREFILL_HOST = ""
_NOTIFY_HOST = ""
_BOOTSTRAP_PORT = 8998
_DECODE_URL = ""
_NOTIFY_TIMEOUT_S = 120.0
_ENGINE_IDS: dict[int, str] = {}


def _bootstrap_addr(payload: dict) -> str:
    port = int(payload.get("remote_bootstrap_port") or _BOOTSTRAP_PORT)
    host = _PREFILL_HOST or "127.0.0.1"
    return f"http://{host}:{port}"


def _discover_prefiller(timeout_s: float = 60.0) -> None:
    """Prefetch the prefiller engine id so Decode need not wait for the response.

    Same /query probe as mooncake_connector_proxy. Engine id and bootstrap
    address are known before the request arrives, so Prefill and Decode run together.
    """
    url = _bootstrap_addr({})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url + "/query", timeout=5.0)
            response.raise_for_status()
            data = response.json()
        except Exception:
            time.sleep(1.0)
            continue
        engine_ids = {
            int(rank): str(entry.get("engine_id"))
            for rank, entry in (data or {}).items()
            if entry.get("engine_id")
        }
        if engine_ids:
            _ENGINE_IDS.clear()
            _ENGINE_IDS.update(engine_ids)
            print(
                f"[layerwise-proxy] prefiller ready at {url} "
                f"dp_size={len(engine_ids)}"
            )
            return
        time.sleep(1.0)
    print(
        f"[layerwise-proxy] prefiller {url}/query unavailable; "
        "decode waits for the /notify callback instead"
    )


def _decode_params_from_local(transfer_id: str) -> dict | None:
    """Build Decode params from the prefetched engine id."""
    engine_id = _ENGINE_IDS.get(0)
    if not engine_id:
        return None
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "transfer_id": transfer_id,
        "remote_engine_id": engine_id,
        "remote_bootstrap_addr": _bootstrap_addr({}),
    }


def _run_prefill(body: dict, transfer_id: str, request_path: str) -> None:
    """Run Prefill in the background. Its KV params are only a fallback if notify is lost."""
    prefill_body = copy.deepcopy(body)
    prefill_body["max_tokens"] = 1
    if "max_completion_tokens" in prefill_body:
        prefill_body["max_completion_tokens"] = 1
    prefill_body["stream"] = False
    prefill_body.pop("stream_options", None)
    params = dict(prefill_body.get("kv_transfer_params") or {})
    # Use the prefiller host so workers inside a container can reach the proxy.
    # The notify endpoint may run on a different host than the prefiller.
    notify_host = _NOTIFY_HOST or _PREFILL_HOST or "127.0.0.1"
    params.update(
        {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "transfer_id": transfer_id,
            "layerwise_notify_url": f"http://{notify_host}:{_PROXY_PORT}/notify",
        }
    )
    prefill_body["kv_transfer_params"] = params
    result: dict = {}
    try:
        response = httpx.post(
            _PREFILL_URL.rstrip("/") + request_path,
            json=prefill_body,
            timeout=None,
        )
        result["status"] = response.status_code
        try:
            result["body"] = response.json()
        except Exception:
            result["body"] = {"raw": response.text}
    except Exception as exc:
        result["error"] = str(exc)
    with _PENDING_LOCK:
        slot = _PENDING.get(transfer_id)
    if slot is not None:
        slot["prefill"] = result
        slot["prefill_done"].set()
        slot["ready"].set()


def _decode_params_from_notify(payload: dict, transfer_id: str) -> dict:
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "transfer_id": transfer_id,
        "remote_engine_id": payload.get("remote_engine_id"),
        "remote_bootstrap_addr": _bootstrap_addr(payload),
    }


def _decode_params_from_prefill(prefill: dict, transfer_id: str) -> dict | None:
    body = prefill.get("body") or {}
    params = body.get("kv_transfer_params") or {}
    if not params.get("remote_engine_id") or not params.get("remote_bootstrap_addr"):
        return None
    merged = dict(params)
    merged["do_remote_prefill"] = True
    merged["do_remote_decode"] = False
    merged["transfer_id"] = params.get("transfer_id") or transfer_id
    return merged


def _prefill_failed(prefill: dict) -> bool:
    status = prefill.get("status", 200)
    return bool(prefill.get("error")) or not 200 <= status < 300


def _wait_for_prefill(slot: dict, transfer_id: str) -> bool:
    """Wake for a notify, an upstream error, or usable fallback metadata."""
    deadline = time.monotonic() + _NOTIFY_TIMEOUT_S
    while True:
        slot["ready"].clear()
        if slot["notify"].is_set():
            return True
        if slot["prefill_done"].is_set():
            prefill = slot.get("prefill") or {}
            if _prefill_failed(prefill) or _decode_params_from_prefill(
                prefill, transfer_id
            ):
                return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        slot["ready"].wait(remaining)


def _forward_decode(body: dict, params: dict, request_path: str):
    decode_body = copy.deepcopy(body)
    decode_body["kv_transfer_params"] = params
    return httpx.stream(
        "POST",
        _DECODE_URL.rstrip("/") + request_path,
        json=decode_body,
        timeout=None,
    )


_PROXY_PORT = 8001


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print("[layerwise-proxy]", fmt % args)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _write(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _write_stream(self, response: httpx.Response) -> None:
        """Forward the decode stream so the client can observe TTFT."""
        self.send_response(response.status_code)
        self.send_header(
            "Content-Type",
            response.headers.get("content-type", "text/event-stream"),
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for chunk in response.iter_bytes():
            if chunk:
                self.wfile.write(chunk)
                self.wfile.flush()

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            data = self._read_json()
        except Exception as exc:
            message = json.dumps({"error": str(exc)}).encode("utf-8")
            self._write(400, message, "application/json")
            return
        if path == "/notify":
            transfer_id = str(data.get("transfer_id") or "")
            with _PENDING_LOCK:
                slot = _PENDING.get(transfer_id)
            if slot is not None and slot.get("payload") is None:
                slot["payload"] = data
                slot["notify"].set()
                slot["ready"].set()
            self._write(204, b"", "text/plain")
            return
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self._write(404, b"{}", "application/json")
            return
        self._handle_request(data, path)

    def _handle_request(self, body: dict, path: str) -> None:
        transfer_id = str(uuid.uuid4())
        slot = {
            "notify": threading.Event(),
            "prefill_done": threading.Event(),
            "ready": threading.Event(),
            "payload": None,
            "prefill": None,
        }
        with _PENDING_LOCK:
            _PENDING[transfer_id] = slot
        try:
            self._serve_request(body, path, transfer_id, slot)
        finally:
            with _PENDING_LOCK:
                _PENDING.pop(transfer_id, None)

    def _serve_request(
        self, body: dict, path: str, transfer_id: str, slot: dict
    ) -> None:
        threading.Thread(
            target=_run_prefill,
            args=(body, transfer_id, path),
            daemon=True,
            name=f"prefill-{transfer_id[:8]}",
        ).start()
        # Engine id is known at startup, so Decode does not wait for Prefill or notify.
        params = _decode_params_from_local(transfer_id)
        if params is not None:
            print(
                "[layerwise-proxy] concurrent decode transfer_id="
                f"{transfer_id} engine={params.get('remote_engine_id')}"
            )
        else:
            notified = _wait_for_prefill(slot, transfer_id)
            if notified and slot.get("payload"):
                params = _decode_params_from_notify(slot["payload"], transfer_id)
                print(
                    "[layerwise-proxy] early decode transfer_id="
                    f"{transfer_id} engine={params.get('remote_engine_id')}"
                )
            else:
                slot["prefill_done"].wait()
                prefill = slot.get("prefill") or {}
                if _prefill_failed(prefill):
                    message = json.dumps(
                        {"error": "Prefill request failed", "prefill": prefill}
                    ).encode("utf-8")
                    self._write(502, message, "application/json")
                    return
                params = _decode_params_from_prefill(prefill, transfer_id)
                print(
                    "[layerwise-proxy] fallback after prefill transfer_id="
                    f"{transfer_id}"
                )
        if not params or not params.get("remote_engine_id"):
            message = json.dumps(
                {
                    "error": "layerwise proxy did not get prefiller engine id",
                    "prefill": slot.get("prefill"),
                }
            ).encode("utf-8")
            self._write(502, message, "application/json")
            return
        try:
            with _forward_decode(body, params, path) as response:
                if body.get("stream"):
                    self._write_stream(response)
                else:
                    content = response.read()
                    self._write(
                        response.status_code,
                        content,
                        response.headers.get("content-type", "application/json"),
                    )
        except Exception as exc:
            content = json.dumps({"error": str(exc)}).encode("utf-8")
            self._write(502, content, "application/json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", nargs=2, metavar=("URL", "BOOTSTRAP_PORT"), required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--notify-host",
        default="",
        help="Host reachable by prefiller workers for the /notify callback.",
    )
    args = parser.parse_args()
    global _PREFILL_URL, _PREFILL_HOST, _BOOTSTRAP_PORT
    global _DECODE_URL, _PROXY_PORT, _NOTIFY_HOST
    _PREFILL_URL = args.prefill[0]
    _BOOTSTRAP_PORT = int(args.prefill[1])
    _DECODE_URL = args.decode
    _PROXY_PORT = args.port
    _NOTIFY_HOST = args.notify_host
    _PREFILL_HOST = urlparse(_PREFILL_URL).hostname or "127.0.0.1"
    _discover_prefiller()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _Handler)
    print(
        "[layerwise-proxy] listen"
        f" 0.0.0.0:{args.port} prefill={_PREFILL_URL} decode={_DECODE_URL}"
        f" engine_ids={len(_ENGINE_IDS)}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
