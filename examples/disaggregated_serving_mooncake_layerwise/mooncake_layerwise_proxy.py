# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Layerwise Mooncake 代理：Prefill 分配 KV 后立刻拉起 Decode。

原来的代理要等 Prefill 的 HTTP 响应才把请求转给 Decode，握手发生在计算结束之后，
按层发送会挤在最后几十毫秒。本代理在 Prefill 请求里带上通知地址，worker 一分配
到 block 就回调 /notify，Decode 在后续层还在计算时完成握手。整包部署不要用它。
"""

from __future__ import annotations

import argparse
import copy
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import httpx

_PENDING: dict[str, dict] = {}
_PENDING_LOCK = threading.Lock()
_PREFILL_URL = ""
_PREFILL_HOST = ""
_BOOTSTRAP_PORT = 8998
_DECODE_URL = ""
_NOTIFY_TIMEOUT_S = 120.0


def _bootstrap_addr(payload: dict) -> str:
    port = int(payload.get("remote_bootstrap_port") or _BOOTSTRAP_PORT)
    host = _PREFILL_HOST or "127.0.0.1"
    return f"http://{host}:{port}"


def _run_prefill(body: dict, transfer_id: str, request_path: str) -> None:
    """后台跑 Prefill。响应里的 kv 参数只在通知丢失时当作退路。"""
    prefill_body = copy.deepcopy(body)
    prefill_body["max_tokens"] = 1
    prefill_body["stream"] = False
    params = dict(prefill_body.get("kv_transfer_params") or {})
    # 用 Prefill 机器的地址，避免 Worker 在容器里访问不到 127.0.0.1 上的代理。
    notify_host = _PREFILL_HOST or "127.0.0.1"
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


def _forward_decode(body: dict, params: dict, request_path: str) -> tuple[int, bytes]:
    decode_body = copy.deepcopy(body)
    decode_body["stream"] = False
    decode_body["kv_transfer_params"] = params
    response = httpx.post(
        _DECODE_URL.rstrip("/") + request_path,
        json=decode_body,
        timeout=None,
    )
    return response.status_code, response.content


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
            "payload": None,
            "prefill": None,
        }
        with _PENDING_LOCK:
            _PENDING[transfer_id] = slot
        threading.Thread(
            target=_run_prefill,
            args=(body, transfer_id, path),
            daemon=True,
            name=f"prefill-{transfer_id[:8]}",
        ).start()
        notified = slot["notify"].wait(_NOTIFY_TIMEOUT_S)
        params = None
        if notified and slot.get("payload"):
            params = _decode_params_from_notify(slot["payload"], transfer_id)
            print(
                "[layerwise-proxy] early decode transfer_id="
                f"{transfer_id} engine={params.get('remote_engine_id')}"
            )
        else:
            slot["prefill_done"].wait()
            params = _decode_params_from_prefill(slot.get("prefill") or {}, transfer_id)
            print("[layerwise-proxy] fallback after prefill transfer_id=" f"{transfer_id}")
        if not params or not params.get("remote_engine_id"):
            message = json.dumps(
                {"error": "layerwise proxy did not get prefiller engine id"}
            ).encode("utf-8")
            self._write(502, message, "application/json")
            return
        try:
            status, content = _forward_decode(body, params, path)
        except Exception as exc:
            content = json.dumps({"error": str(exc)}).encode("utf-8")
            status = 502
        self._write(status, content, "application/json")
        with _PENDING_LOCK:
            _PENDING.pop(transfer_id, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", nargs=2, metavar=("URL", "BOOTSTRAP_PORT"), required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    global _PREFILL_URL, _PREFILL_HOST, _BOOTSTRAP_PORT, _DECODE_URL, _PROXY_PORT
    _PREFILL_URL = args.prefill[0]
    _BOOTSTRAP_PORT = int(args.prefill[1])
    _DECODE_URL = args.decode
    _PROXY_PORT = args.port
    _PREFILL_HOST = urlparse(_PREFILL_URL).hostname or "127.0.0.1"
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _Handler)
    print(
        "[layerwise-proxy] listen"
        f" 0.0.0.0:{args.port} prefill={_PREFILL_URL} decode={_DECODE_URL}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
