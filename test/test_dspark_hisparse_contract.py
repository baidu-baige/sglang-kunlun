import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin
from sglang_kunlun.hooks.speculative import dspark_precision


class _Stream:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def wait_stream(self, stream):
        self.events.append(("wait_stream", self.name, stream.name))


class _DeviceModule:
    def __init__(self, events):
        self.events = events
        self.current = _Stream("schedule", events)

    def current_stream(self):
        self.events.append(("current_stream", self.current.name))
        return self.current

    class _Context:
        def __init__(self, stream, events):
            self.stream = stream
            self.events = events

        def __enter__(self):
            self.events.append(("enter_stream", self.stream.name))

        def __exit__(self, *_):
            self.events.append(("exit_stream", self.stream.name))

    def stream(self, stream):
        return self._Context(stream, self.events)


class TestDsparkHisparseContract(unittest.TestCase):
    def _runner(self, host_alloc):
        events = []
        backup_stream = _Stream("backup", events)
        producer_stream = _Stream("producer", events)
        source_inputs = []
        translated_inputs = []
        backup_calls = []

        allocator = MagicMock()

        def get_last(loc):
            source_inputs.append(loc.detach().cpu().clone())
            return loc + 100

        allocator.get_last_loc_compressed.side_effect = get_last
        device_pool = MagicMock()

        def translate(loc):
            translated_inputs.append(loc.detach().cpu().clone())
            return loc + 1000

        device_pool._translate_loc_to_hisparse_device.side_effect = translate
        host_pool = SimpleNamespace(
            alloc_paged_token_slots=host_alloc,
            backup_from_device_all_layer=MagicMock(
                side_effect=lambda host, host_locs, source_locs, io_backend: backup_calls.append(
                    (host, host_locs.detach().cpu().clone(), source_locs.detach().cpu().clone(), io_backend)
                )
            ),
        )
        done_event = MagicMock()
        done_event.record.side_effect = lambda: events.append(("event_record",))
        coordinator = SimpleNamespace(
            is_dsv4_hisparse=True,
            compress_ratio=4,
            mem_pool_device=device_pool,
            mem_pool_host=host_pool,
            req_to_host_pool=torch.zeros((1, 16), dtype=torch.int64),
            req_to_host_pool_allocated_len=torch.zeros(1, dtype=torch.int64),
            wait_for_pending_backup=MagicMock(
                side_effect=lambda: events.append(("wait_for_pending_backup",))
            ),
            decode_backup_stream=backup_stream,
            decode_producer_stream=producer_stream,
            _backup_done_event=done_event,
            _has_pending_backup=False,
        )
        runner = SimpleNamespace(
            hisparse_coordinator=coordinator,
            token_to_kv_pool_allocator=allocator,
            device="cpu",
        )
        return runner, _DeviceModule(events), events, source_inputs, translated_inputs, backup_calls

    def test_backups_committed_ratio_four_positions(self):
        calls = []

        def alloc(req, allocated, req_idx, position, count):
            calls.append((req_idx, position, count))
            return torch.tensor([500 + position], dtype=torch.int64)

        runner, device, events, source, translated, backups = self._runner(alloc)
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0]),
            out_cache_loc=torch.arange(12),
        )
        with patch.object(dspark_precision.torch, "get_device_module", return_value=device):
            dspark_precision._backup_committed_c4_positions(
                batch=batch,
                prefix_lens=torch.tensor([0]),
                new_seq_lens=torch.tensor([12]),
                num_draft=12,
                model_runner=runner,
            )

        self.assertEqual(calls, [(0, 0, 1), (0, 1, 1), (0, 2, 1)])
        self.assertEqual([x.item() for x in source], [3, 7, 11])
        self.assertEqual([x.item() for x in translated], [103, 107, 111])
        self.assertEqual(len(backups), 1)
        _, host_locs, source_locs, io_backend = backups[0]
        self.assertEqual(host_locs.tolist(), [500, 501, 502])
        self.assertEqual(source_locs.tolist(), [1103, 1107, 1111])
        self.assertEqual(io_backend, "kernel")
        self.assertIn(("wait_for_pending_backup",), events)
        self.assertIn(("wait_stream", "backup", "schedule"), events)
        self.assertIn(("wait_stream", "backup", "producer"), events)
        self.assertIn(("event_record",), events)
        self.assertTrue(runner.hisparse_coordinator._has_pending_backup)

    def test_partial_acceptance_backs_up_only_committed_boundary(self):
        calls = []

        def alloc(req, allocated, req_idx, position, count):
            calls.append((req_idx, position, count))
            return torch.tensor([700 + position], dtype=torch.int64)

        runner, device, _, source, _, backups = self._runner(alloc)
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0]),
            out_cache_loc=torch.arange(4, 10),
        )
        with patch.object(dspark_precision.torch, "get_device_module", return_value=device):
            dspark_precision._backup_committed_c4_positions(
                batch=batch,
                prefix_lens=torch.tensor([4]),
                new_seq_lens=torch.tensor([9]),
                num_draft=6,
                model_runner=runner,
            )

        self.assertEqual(calls, [(0, 1, 1)])
        self.assertEqual([x.item() for x in source], [7])
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0][1].tolist(), [701])
        self.assertEqual(backups[0][2].tolist(), [1107])

    def test_host_pool_reuses_pages_and_returns_stable_slots(self):
        class TinyHostPool(HiSparseHostPoolMixin):
            page_size = 4

            def __init__(self):
                self.alloc_calls = []

            def alloc(self, size):
                self.alloc_calls.append(size)
                return torch.arange(size, dtype=torch.int64)

        pool = TinyHostPool()
        req_to_host_pool = torch.full((1, 8), -1, dtype=torch.int64)
        allocated_len = torch.zeros(1, dtype=torch.int64)
        first = pool.alloc_paged_token_slots(req_to_host_pool, allocated_len, 0, 0, 1)
        second = pool.alloc_paged_token_slots(req_to_host_pool, allocated_len, 0, 0, 1)
        third = pool.alloc_paged_token_slots(req_to_host_pool, allocated_len, 0, 1, 1)

        self.assertEqual(pool.alloc_calls, [4])
        self.assertEqual(int(allocated_len[0]), 4)
        self.assertEqual(first.tolist(), [0])
        self.assertEqual(second.tolist(), [0])
        self.assertEqual(third.tolist(), [1])
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
