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
from unittest.mock import Mock, patch


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
            "_schedule_layerwise_flush", "_flush_scheduled_layers",
            "_layerwise_session_key", "_register_layerwise_session",
            "_layerwise_flush", "_flush_one_layerwise_session", "start_load_kv",
            "_build_transfer_params",
        },
    }
    body = [ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    )]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_apply_alloc_to_layerwise_sessions", "_region_belongs_to_layer",
            "_can_coalesce_block_transfers", "_validate_nonnegative_block_ids",
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
        "MambaSpec": type("MambaSpec", (), {}),
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
        self.builds = []
        self.events = []
        self.notifies = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(self.executor.shutdown, wait=True)
        worker = runtime.MooncakeConnectorWorker()
        worker.layerwise_enabled = True
        worker.is_kv_producer = True
        worker.is_kv_consumer = False
        worker._layerwise_lock = threading.Lock()
        worker._layerwise_epoch = 0
        worker._layerwise_computed = {}
        worker._layerwise_flush_scheduled = False
        worker._layerwise_flush_lock = asyncio.Lock()
        worker._layerwise_alloc_blocks = {}
        worker._layerwise_final_blocks = {}
        worker._layerwise_sessions = {}
        worker.sender_loop = asyncio.get_running_loop()
        worker._sender_executor = self.executor
        worker._is_canonical_pcp_kv_replica = lambda: True
        worker._notify_layerwise_proxy = lambda payload: self.notifies.append(payload)

        async def build_params(_reqs, _meta, local, _remote):
            layers = [region.layer_index for region in local]
            self.builds.append(layers)
            return layers, layers, [64] * len(layers), [], None

        def send(_event, _remote_session, src, _dst, _lengths):
            self.events.append(_event)
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

    def _handshake(self, regions=None):
        send_meta = SimpleNamespace(
            transfer_id="xfer", local_block_ids=[], p_req_id="req",
            ttft_send_start_logged=False,
        )
        meta = SimpleNamespace(
            remote_hostname="decoder", remote_port=1234,
            remote_tp_rank=0, remote_pp_size=1,
        )
        if regions is None:
            regions = [SimpleNamespace(
                layer_name=f"model.layers.{i}.self_attn", layer_index=i,
            ) for i in range(16)]
        self.worker._register_layerwise_session(
            "decode-req", send_meta, meta, regions, regions,
        )
        return next(iter(self.worker._layerwise_sessions.values()))

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
        self.assertEqual(set(self.worker._layerwise_computed), {1})
        self.assertEqual(len(self.notifies), 1)
        self.assertEqual(self.worker._layerwise_final_blocks, {})

    async def test_sync_load_order_also_initializes_each_forward_once(self):
        for epoch in (1, 2):
            self.connector.bind_connector_metadata(self.metadata)
            self.connector.start_load_kv(None)
            self.connector.save_kv_layer("model.layers.0.self_attn", None, None)
            self.assertEqual(self.worker._layerwise_epoch, epoch)
            self.assertEqual(
                self.worker._layerwise_computed[epoch][-1][0],
                "model.layers.0.self_attn",
            )

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
        self.assertEqual(self.worker._layerwise_computed, {})
        self.assertEqual(self.notifies, [])

    async def test_callbacks_coalesce_without_losing_a_later_layer(self):
        self.connector.bind_connector_metadata(self.metadata)
        self._handshake()
        await self.worker._flush_scheduled_layers()
        with patch.object(
            asyncio, "run_coroutine_threadsafe", wraps=asyncio.run_coroutine_threadsafe
        ) as schedule:
            for i in range(8):
                self.connector.save_kv_layer(f"model.layers.{i}.self_attn", None, None)
            self.assertEqual(schedule.call_count, 1)
        await self._compute_layers(8, 16)
        self.assertEqual(
            sorted(i for batch in self.batches for i in batch), list(range(16))
        )

    async def test_each_hook_is_parsed_once_and_descriptors_are_built_per_group(self):
        self.connector.bind_connector_metadata(self.metadata)
        regions = [SimpleNamespace(
            layer_name=f"model.layers.{i}.{suffix}", layer_index=i,
        ) for i in range(78) for suffix in ("self_attn", "self_attn.indexer")]
        session = self._handshake(regions)
        parse = Mock(wraps=runtime.extract_layer_index)
        with patch.dict(
            self.worker._flush_one_layerwise_session.__func__.__globals__,
            {"extract_layer_index": parse},
        ):
            for i in range(78):
                await self._compute_layers(i, i + 1)
            self.worker._layerwise_final_blocks["xfer"] = [[1, 2]]
            await self._compute_layers(78, 78)
            self.assertEqual(parse.call_count, 78)
        self.assertEqual(len(self.builds), 10)
        self.assertEqual(len(self.batches), 10)
        self.assertEqual(
            [i for batch in self.batches for i in batch],
            [i for i in range(78) for _ in range(2)],
        )
        self.assertEqual(session["computed_cursor"], 78)
        self.assertEqual(session["pending_events"], {})

    async def test_group_waits_for_last_callback_instead_of_highest_layer(self):
        self.connector.bind_connector_metadata(self.metadata)
        session = self._handshake()
        events = [object() for _ in range(9)]
        self.worker._layerwise_computed[1] = [
            (f"model.layers.{i}.self_attn", events[i]) for i in reversed(range(8))
        ] + [("model.layers.0.self_attn", events[8])]
        await self._compute_layers(0, 0)
        self.assertEqual(self.batches, [list(range(8))])
        self.assertEqual(self.events, [events[8]])
        self.assertEqual(session["computed_cursor"], 9)

    async def test_handshake_before_allocation_gets_the_new_forward_epoch(self):
        self.worker._layerwise_alloc_blocks.clear()
        session = self._handshake()
        self.assertEqual(session["epoch"], 0)
        self.connector.bind_connector_metadata(self.metadata)
        await self._compute_layers(0, 8)
        self.assertEqual(session["epoch"], 1)
        self.assertEqual(self.batches, [list(range(8))])

    async def test_final_metadata_flushes_hooks_and_resends_changed_blocks(self):
        self.connector.bind_connector_metadata(self.metadata)
        session = self._handshake()
        await self._compute_layers(0, 8)
        self.worker._layerwise_final_blocks["xfer"] = [[1, 2, 3]]
        await self._compute_layers(8, 8)
        await self._compute_layers(8, 8)
        self.assertEqual(
            self.batches, [list(range(8)), list(range(8, 16)), list(range(16))]
        )
        self.assertEqual(session["send_meta"].local_block_ids, [[1, 2, 3]])
        self.assertTrue(session["full_resent"])

    async def test_concurrent_flushes_do_not_submit_duplicate_layers(self):
        self.connector.bind_connector_metadata(self.metadata)
        self._handshake()
        build_params = self.worker._build_transfer_params

        async def delayed_build(*args):
            await asyncio.sleep(0)
            return await build_params(*args)

        self.worker._build_transfer_params = delayed_build
        self.worker._layerwise_computed[1] = [
            (f"model.layers.{i}.self_attn", object()) for i in range(8)
        ]
        await asyncio.gather(
            self.worker._layerwise_flush(), self.worker._layerwise_flush()
        )
        await self._compute_layers(0, 0)
        self.assertEqual(self.batches, [list(range(8))])

    async def test_grouped_descriptor_build_preserves_each_kv_group_address(self):
        worker = self.worker
        worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=object()) for _ in range(2)
        ])
        worker._logical_to_kernel_block_ids = lambda groups: groups
        worker._get_sender_transfer_plan = lambda **kwargs: (True, 0, 0, 8)
        session = self._handshake()
        session["send_meta"].local_block_ids = [[1, 2], [3]]
        meta = session["meta"]
        meta.req_blocks = {"decode-req": ("xfer", [[4, 5], [6]])}
        meta.remote_tp_size = 1
        local = [SimpleNamespace(
            layer_name=f"model.layers.{i}.self_attn", layer_index=i,
            base_addr=1000 * (i + 1), block_len=8, kv_block_len=8,
            group_index=i % 2, num_blocks=16,
        ) for i in range(8)]
        remote = [SimpleNamespace(**{
            **vars(region), "base_addr": region.base_addr + 10000,
        }) for region in local]
        build = runtime.MooncakeConnectorWorker._build_transfer_params
        args = ([("decode-req", session["send_meta"])], meta)
        namespace = build.__globals__
        with patch.dict(namespace, {
            # Both groups are contiguous in this fixture.
            "group_concurrent_contiguous": lambda src, dst: ([src], [dst]),
        }):
            grouped = await build(worker, *args, local, remote)
            individual = [await build(worker, *args, [a], [b])
                          for a, b in zip(local, remote)]
        for column in range(3):
            self.assertEqual(
                grouped[column],
                [value for result in individual for value in result[column]],
            )
        self.assertEqual(grouped[3:], ([], None))
        self.assertEqual(grouped[0][:2], [1008, 2024])
        self.assertEqual(grouped[1][:2], [11032, 12048])

    async def test_build_error_does_not_submit_the_group(self):
        self.connector.bind_connector_metadata(self.metadata)
        session = self._handshake()

        async def fail_build(*_args):
            return [], [], [], ["decode-req"], "invalid blocks"

        self.worker._build_transfer_params = fail_build
        await self._compute_layers(0, 8)
        self.assertEqual(self.batches, [])
        self.assertEqual(session["errors"], ["invalid blocks"])


if __name__ == "__main__":
    unittest.main()
