# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import copy
import importlib.util
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx


_PROXY_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/disaggregated_serving_mooncake_layerwise/mooncake_layerwise_proxy.py"
)
_SPEC = importlib.util.spec_from_file_location("layerwise_proxy_test_target", _PROXY_PATH)
proxy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(proxy)


def _slot():
    return {
        "notify": threading.Event(),
        "prefill_done": threading.Event(),
        "ready": threading.Event(),
        "payload": None,
        "prefill": None,
    }


class LayerwiseProxyTest(unittest.TestCase):
    def setUp(self):
        proxy._PENDING.clear()

    def tearDown(self):
        proxy._PENDING.clear()

    def test_benchmark_prefill_is_valid_and_client_request_is_preserved(self):
        for limit in ("max_tokens", "max_completion_tokens"):
            with self.subTest(limit=limit):
                body = {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "test"}],
                    limit: 16,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                original = copy.deepcopy(body)
                slot = _slot()
                proxy._PENDING["test-transfer"] = slot
                response = httpx.Response(200, json={"choices": []})
                with patch.object(proxy.httpx, "post", return_value=response) as post:
                    proxy._run_prefill(body, "test-transfer", "/v1/chat/completions")
                sent = post.call_args.kwargs["json"]
                self.assertFalse(sent["stream"])
                self.assertNotIn("stream_options", sent)
                self.assertEqual(sent[limit], 1)
                self.assertEqual(body, original)
                self.assertTrue(slot["prefill_done"].is_set())
                self.assertTrue(slot["ready"].is_set())
                with patch.object(proxy.httpx, "stream") as stream:
                    proxy._forward_decode(body, {"transfer_id": "test-transfer"},
                                          "/v1/chat/completions")
                decode_body = stream.call_args.kwargs["json"]
                self.assertTrue(decode_body["stream"])
                self.assertEqual(decode_body[limit], 16)
                self.assertEqual(decode_body["stream_options"], body["stream_options"])

    def test_prefill_error_wakes_without_notify_timeout(self):
        slot = _slot()
        slot["prefill"] = {"status": 400, "body": {"error": "invalid request"}}
        slot["prefill_done"].set()
        slot["ready"] = Mock()
        self.assertFalse(proxy._wait_for_prefill(slot, "test-transfer"))
        slot["ready"].wait.assert_not_called()

    def test_bulk_fallback_wakes_on_metadata(self):
        slot = _slot()
        slot["prefill"] = {
            "status": 200,
            "body": {"kv_transfer_params": {
                "remote_engine_id": "test-engine",
                "remote_bootstrap_addr": "http://prefill:8998",
            }},
        }
        slot["prefill_done"].set()
        slot["ready"] = Mock()
        self.assertFalse(proxy._wait_for_prefill(slot, "test-transfer"))
        slot["ready"].wait.assert_not_called()

    def test_prefill_without_metadata_still_waits_for_late_notify(self):
        slot = _slot()
        slot["prefill"] = {"status": 200, "body": {"kv_transfer_params": None}}
        slot["prefill_done"].set()
        slot["ready"] = Mock()

        def notify(_timeout):
            slot["payload"] = {"remote_engine_id": "test-engine"}
            slot["notify"].set()

        slot["ready"].wait.side_effect = notify
        self.assertTrue(proxy._wait_for_prefill(slot, "test-transfer"))
        slot["ready"].wait.assert_called_once()

    def test_early_notify_does_not_wait_for_prefill_completion(self):
        slot = _slot()
        slot["payload"] = {"remote_engine_id": "test-engine"}
        slot["notify"].set()
        slot["ready"] = Mock()
        self.assertTrue(proxy._wait_for_prefill(slot, "test-transfer"))
        self.assertFalse(slot["prefill_done"].is_set())
        slot["ready"].wait.assert_not_called()

    def test_upstream_failures_are_visible_and_pending_state_is_cleaned(self):
        cases = (
            httpx.Response(400, json={"error": "stream options require streaming"}),
            httpx.ConnectError("prefiller unreachable"),
        )
        for result in cases:
            with self.subTest(result=result):
                handler = object.__new__(proxy._Handler)
                handler._write = Mock()
                # Run the producer synchronously so a missing wake fails immediately.
                def start_thread(*, target, args, **kwargs):
                    return Mock(start=lambda: target(*args))

                post_kwargs = ({"side_effect": result} if isinstance(result, Exception)
                               else {"return_value": result})
                with (
                    patch.object(proxy.threading, "Thread", side_effect=start_thread),
                    patch.object(proxy.httpx, "post", **post_kwargs),
                    patch.object(proxy, "_forward_decode") as decode,
                ):
                    handler._handle_request({"stream": True}, "/v1/chat/completions")
                status, payload, _ = handler._write.call_args.args
                detail = json.loads(payload)
                self.assertEqual(status, 502)
                self.assertEqual(detail["error"], "Prefill request failed")
                if isinstance(result, Exception):
                    self.assertEqual(detail["prefill"]["error"], str(result))
                else:
                    self.assertEqual(detail["prefill"]["status"], 400)
                    self.assertEqual(detail["prefill"]["body"], result.json())
                decode.assert_not_called()
                self.assertFalse(proxy._PENDING)


if __name__ == "__main__":
    unittest.main()
