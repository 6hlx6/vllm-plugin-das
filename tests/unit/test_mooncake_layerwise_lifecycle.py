# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import ast
import asyncio
import logging
import re
import threading
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "vllm_hcu/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
)


class _BaseConnector:
    def bind_connector_metadata(self, metadata):
        self._connector_metadata = metadata


def _load_lifecycle():
    """Run production lifecycle methods without GPU or RDMA dependencies."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    selected = {
        "MooncakeConnectorMetadata": {"__init__"},
        "MooncakeConnector": {
            "bind_connector_metadata", "start_load_kv", "save_kv_layer",
        },
        "MooncakeConnectorWorker": {
            "_begin_layerwise_forward", "notify_layer_computed",
            "_layerwise_session_key", "_register_layerwise_session",
            "_layerwise_flush", "_flush_one_layerwise_session", "start_load_kv",
        },
    }
    body = [ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    )]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_apply_alloc_to_layerwise_sessions", "_region_belongs_to_layer",
        }:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in selected:
            node.bases = [ast.Name(id="_BaseConnector", ctx=ast.Load())]
            node.keywords = []
            node.body = [method for method in node.body if (
                isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name in selected[node.name]
            )]
            body.append(node)
    namespace = {
        "_BaseConnector": _BaseConnector,
        "asyncio": asyncio,
        "defaultdict": defaultdict,
        "logger": logging.getLogger(__name__),
        "extract_layer_index": lambda name: int(re.search(r"layers\.(\d+)", name)[1]),
        "_record_kv_ready_event": object,
        "_LAYERWISE_SEND_GROUP": 8,
        "log_ttft_event": lambda *_args, **_kwargs: None,
    }
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(_SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace)


runtime = _load_lifecycle()


class LayerwiseLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.batches = []
        self.notifies = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(self.executor.shutdown, wait=True)
        worker = runtime.MooncakeConnectorWorker()
        worker.layerwise_enabled = True
        worker.is_kv_producer = True
        worker.is_kv_consumer = False
        worker._layerwise_lock = threading.Lock()
        worker._layerwise_epoch = 0
        worker._layerwise_computed = []
        worker._layerwise_alloc_blocks = {}
        worker._layerwise_final_blocks = {}
        worker._layerwise_sessions = {}
        worker.sender_loop = asyncio.get_running_loop()
        worker._sender_executor = self.executor
        worker._is_canonical_pcp_kv_replica = lambda: True
        worker._notify_layerwise_proxy = lambda payload: self.notifies.append(payload)

        async def build_params(_reqs, _meta, local, _remote):
            layers = [region.layer_index for region in local]
            return layers, layers, [64] * len(layers), [], None

        def send(_event, _remote_session, src, _dst, _lengths):
            self.batches.append(list(src))
            return 0

        worker._build_transfer_params = build_params
        worker._send_layer_blocks = send
        self.worker = worker
        self.connector = runtime.MooncakeConnector()
        self.connector.connector_worker = worker
        self.metadata = runtime.MooncakeConnectorMetadata()
        self.metadata.layerwise_alloc["req"] = ("xfer", [[1, 2]])
        self.metadata.layerwise_notify["xfer"] = "http://proxy/notify"

    def _handshake(self):
        send_meta = SimpleNamespace(
            transfer_id="xfer", local_block_ids=[], p_req_id="req",
            ttft_send_start_logged=False,
        )
        meta = SimpleNamespace(
            remote_hostname="decoder", remote_port=1234,
            remote_tp_rank=0, remote_pp_size=1,
        )
        regions = [SimpleNamespace(
            layer_name=f"model.layers.{i}.self_attn", layer_index=i,
        ) for i in range(16)]
        self.worker._register_layerwise_session(
            "decode-req", send_meta, meta, regions, regions,
        )

    async def _compute_layers(self, start, end):
        for i in range(start, end):
            self.connector.save_kv_layer(f"model.layers.{i}.self_attn", None, None)
        await self.worker._layerwise_flush()
        for session in self.worker._layerwise_sessions.values():
            for future in session["futures"]:
                await future

    async def test_deferred_load_preserves_two_batches_sent_during_forward(self):
        # MRV2 binds first, runs attention, then starts non-sync loads.
        self.connector.bind_connector_metadata(self.metadata)
        self.assertIs(self.connector._connector_metadata, self.metadata)
        self.assertEqual(self.worker._layerwise_epoch, 1)
        self.assertEqual(self.notifies, [self.metadata.layerwise_notify])
        self._handshake()
        await self._compute_layers(0, 8)
        self.assertEqual(self.batches, [list(range(8))])
        await self._compute_layers(8, 16)
        self.assertEqual(self.batches, [list(range(8)), list(range(8, 16))])

        self.connector.start_load_kv(None)
        self.assertEqual(self.worker._layerwise_epoch, 1)
        self.assertEqual({e[0] for e in self.worker._layerwise_computed}, {1})
        self.assertEqual(len(self.notifies), 1)
        self.assertEqual(self.worker._layerwise_final_blocks, {})

    async def test_sync_load_order_also_initializes_each_forward_once(self):
        for epoch in (1, 2):
            self.connector.bind_connector_metadata(self.metadata)
            self.connector.start_load_kv(None)
            self.connector.save_kv_layer("model.layers.0.self_attn", None, None)
            self.assertEqual(self.worker._layerwise_epoch, epoch)
            self.assertEqual(self.worker._layerwise_computed[-1][0], epoch)

    async def test_late_handshake_can_send_computed_layers_before_final_metadata(self):
        self.connector.bind_connector_metadata(self.metadata)
        await self._compute_layers(0, 16)
        self.assertEqual(self.batches, [])
        self.connector.start_load_kv(None)
        self._handshake()
        await self.worker._layerwise_flush()
        for session in self.worker._layerwise_sessions.values():
            for future in session["futures"]:
                await future
            self.assertFalse(session.get("prefill_done", False))
        self.assertEqual(self.batches, [list(range(16))])

    async def test_disabled_layerwise_does_not_initialize_or_record_layers(self):
        self.worker.layerwise_enabled = False
        self.connector.bind_connector_metadata(self.metadata)
        self.connector.save_kv_layer("model.layers.0.self_attn", None, None)
        self.connector.start_load_kv(None)
        self.assertIs(self.connector._connector_metadata, self.metadata)
        self.assertEqual(self.worker._layerwise_epoch, 0)
        self.assertEqual(self.worker._layerwise_computed, [])
        self.assertEqual(self.notifies, [])


if __name__ == "__main__":
    unittest.main()
