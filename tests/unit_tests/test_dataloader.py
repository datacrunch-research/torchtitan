# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.text_datasets import (
    CachedBatchDataset,
    HFDataSource,
    HuggingFaceTextDataLoader,
    InterleavedHuggingFaceTextDataLoader,
)


class DummyDataset(IterableDataset):
    """A simple dummy dataset for testing."""

    def __iter__(self):
        for i in range(100):
            yield {"input": i}, i


class DummyTokenizer(BaseTokenizer):
    """A dummy tokenizer for testing that implements BaseTokenizer interface."""

    def __init__(self):
        super().__init__()
        self.eos_id = 2
        self.bos_id = 1

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        # Simple encoding: convert each character to its ASCII value
        tokens = [ord(c) for c in text]
        if add_bos:
            tokens.insert(0, self.bos_id)  # BOS token
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        # Simple decoding: convert ASCII values back to characters
        return "".join(chr(t) for t in token_ids if t > 2)

    def get_vocab_size(self) -> int:
        return 256  # ASCII range


class TestParallelAwareDataloader(unittest.TestCase):
    def test_dataloader_yields_correct_batches(self):
        """Test that the dataloader correctly yields batched data from the dataset."""
        dataset = DummyDataset()
        batch_size = 4

        dataloader = ParallelAwareDataloader(
            dataset,
            dp_rank=0,
            dp_world_size=1,
            batch_size=batch_size,
        )

        batches = list(dataloader)

        # DummyDataset yields 100 items, so we expect 25 batches of size 4
        self.assertEqual(len(batches), 25)

        # Check first batch structure and values
        first_batch_input, first_batch_label = batches[0]
        self.assertEqual(len(first_batch_input["input"]), batch_size)
        self.assertEqual(len(first_batch_label), batch_size)

        # Verify first batch contains expected values (0, 1, 2, 3)
        self.assertEqual(first_batch_input["input"].tolist(), [0, 1, 2, 3])
        self.assertEqual(first_batch_label.tolist(), [0, 1, 2, 3])

        # Check last batch
        last_batch_input, last_batch_label = batches[-1]
        self.assertEqual(last_batch_input["input"].tolist(), [96, 97, 98, 99])
        self.assertEqual(last_batch_label.tolist(), [96, 97, 98, 99])

    def test_load_state_dict_missing_rank_warning_includes_rank_id(self):
        """The missing-rank warning must interpolate the actual rank key."""
        dataloader = ParallelAwareDataloader(
            DummyDataset(),
            dp_rank=0,
            dp_world_size=1,
            batch_size=4,
        )
        # Non-empty state that lacks this rank's key hits the warning branch.
        state_dict = {"dp_rank_1": b"", "world_size": 1}

        with self.assertLogs(level="WARNING") as cm:
            dataloader.load_state_dict(state_dict)

        output = "\n".join(cm.output)
        self.assertIn(dataloader._rank_id, output)
        self.assertNotIn("{self._rank_id}", output)

    def test_validate_kwargs_rejects_invalid_kwargs(self):
        """Test that passing invalid kwargs raises ValueError."""
        dataset = DummyDataset()

        with self.assertRaises(ValueError) as context:
            ParallelAwareDataloader(
                dataset,
                dp_rank=0,
                dp_world_size=1,
                invalid_arg=42,
            )

        self.assertIn("Invalid dataloader kwargs", str(context.exception))
        self.assertIn("invalid_arg", str(context.exception))

    def test_config_batch_size_overwritten_by_explicit_batch_size(self):
        """Test that batch_size in config kwargs is overwritten by explicit batch_size."""
        dataset = DummyDataset()

        config_kwargs = {"batch_size": 2, "num_workers": 0}

        explicit_batch_size = 8

        # Merge kwargs with explicit args taking precedence (same pattern as in dataset files)
        dataloader_kwargs = {
            **config_kwargs,
            "batch_size": explicit_batch_size,
        }

        dataloader = ParallelAwareDataloader(
            dataset,
            dp_rank=0,
            dp_world_size=1,
            **dataloader_kwargs,
        )

        # Verify that batch_size is the explicit one, not the config one
        self.assertEqual(dataloader.batch_size, explicit_batch_size)

    def test_build_dataloader_with_trainer_config(self):
        """Verify batch_size from training.local_batch_size is correctly used."""
        tokenizer = DummyTokenizer()

        dl_config = HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
            num_workers=2,
        )

        dataloader = HuggingFaceTextDataLoader(
            dl_config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=512,
            local_batch_size=8,
        )

        self.assertEqual(dataloader.batch_size, 8)
        self.assertEqual(dataloader.num_workers, 2)

    def test_positions_matching_sequences(self):
        tokenizer = DummyTokenizer()

        dl_config = HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
            num_workers=0,
            infinite=False,
        )

        dataloader = HuggingFaceTextDataLoader(
            dl_config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=(seq_len := 512),
            local_batch_size=8,
        )

        for batch, _ in zip(map(lambda x: x[0], dataloader), range(10)):
            batch_input_ids = batch["input"]
            batch_positions = batch["positions"]
            for input_ids, positions in zip(batch_input_ids, batch_positions):
                for i, (tok, pos) in enumerate(zip(input_ids, positions)):
                    # pos is less then seq_len
                    self.assertLess(pos.item(), seq_len)
                    self.assertGreaterEqual(pos.item(), 0)
                    if i == 0:
                        # First token should always have position 0
                        self.assertEqual(pos.item(), 0)
                    if i > 0 and pos.item() > 0:
                        # Position should increment by 1 for each subsequent token
                        self.assertEqual(pos.item(), positions[i - 1].item() + 1)
                    if tok == tokenizer.eos_id and i < len(input_ids) - 1:
                        # After EOS, positions should reset to 0
                        self.assertEqual(positions[i + 1].item(), 0)
                    if tok == tokenizer.bos_id and i > 0:
                        # BOS token should have position 0
                        self.assertEqual(pos.item(), 0)


class TestInterleavedHuggingFaceTextDataLoader(unittest.TestCase):
    def _make_config(self, **kwargs) -> InterleavedHuggingFaceTextDataLoader.Config:
        defaults = dict(
            sources=[
                HFDataSource(dataset="c4_test", weight=1.0, infinite=False),
                HFDataSource(dataset="c4_test", weight=1.0, infinite=False),
            ],
            seed=42,
            num_workers=0,
        )
        defaults.update(kwargs)
        return InterleavedHuggingFaceTextDataLoader.Config(**defaults)

    def test_rejects_empty_sources(self):
        with self.assertRaises(ValueError) as ctx:
            InterleavedHuggingFaceTextDataLoader.Config(sources=[], seed=42)
        self.assertIn("At least one source", str(ctx.exception))

    def test_rejects_mixed_infinite(self):
        with self.assertRaises(ValueError) as ctx:
            InterleavedHuggingFaceTextDataLoader.Config(
                sources=[
                    HFDataSource(dataset="c4_test", weight=1.0, infinite=True),
                    HFDataSource(dataset="c4_test", weight=1.0, infinite=False),
                ],
                seed=42,
            )
        self.assertIn("infinite", str(ctx.exception))

    def test_construction_batch_size_and_num_workers(self):
        """Verify local_batch_size and num_workers are correctly plumbed through."""
        config = self._make_config(num_workers=2)
        dataloader = InterleavedHuggingFaceTextDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=DummyTokenizer(),
            seq_len=512,
            local_batch_size=4,
        )
        self.assertEqual(dataloader.batch_size, 4)
        self.assertEqual(dataloader.num_workers, 2)

    def test_yields_input_and_positions_keys(self):
        """Batches must contain 'input' and 'positions' keys, matching single-source format."""
        config = self._make_config()
        dataloader = InterleavedHuggingFaceTextDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=DummyTokenizer(),
            seq_len=512,
            local_batch_size=2,
        )
        batch_input, batch_label = next(iter(dataloader))
        self.assertIn("input", batch_input)
        self.assertIn("positions", batch_input)
        self.assertEqual(batch_input["input"].shape[0], 2)  # batch size
        self.assertEqual(batch_input["input"].shape[1], 512)  # seq_len

    def test_single_source_equivalent_to_huggingfacetextdataloader(self):
        """A single-source interleaved dataloader must produce the same batch
        shape as HuggingFaceTextDataLoader with the same config."""
        tokenizer = DummyTokenizer()
        seq_len = 512
        local_batch_size = 4

        single_dl = HuggingFaceTextDataLoader(
            HuggingFaceTextDataLoader.Config(
                dataset="c4_test", num_workers=0, infinite=False
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
        )

        interleaved_dl = InterleavedHuggingFaceTextDataLoader(
            self._make_config(
                sources=[HFDataSource(dataset="c4_test", weight=1.0, infinite=False)],
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
        )

        single_batch_input, _ = next(iter(single_dl))
        interleaved_batch_input, _ = next(iter(interleaved_dl))

        self.assertEqual(
            single_batch_input["input"].shape,
            interleaved_batch_input["input"].shape,
        )
        self.assertEqual(
            single_batch_input["positions"].shape,
            interleaved_batch_input["positions"].shape,
        )


class TestCachedBatchDataset(unittest.TestCase):
    """Cover repeating a fixed prefix of the dataset (config.num_cached_batches)."""

    def _make_dataloader(self, num_cached_batches: int, local_batch_size: int = 2):
        return HuggingFaceTextDataLoader(
            HuggingFaceTextDataLoader.Config(
                dataset="c4_test",
                num_workers=0,
                infinite=False,
                num_cached_batches=num_cached_batches,
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=DummyTokenizer(),
            seq_len=128,
            local_batch_size=local_batch_size,
        )

    def test_batches_repeat_with_period_num_cached_batches(self):
        num_cached_batches = 2
        dataloader = self._make_dataloader(num_cached_batches)

        batches = []
        for i, batch in enumerate(dataloader):
            batches.append(batch)
            if len(batches) == 3 * num_cached_batches:
                break

        for i in range(num_cached_batches, len(batches)):
            inputs, labels = batches[i]
            ref_inputs, ref_labels = batches[i - num_cached_batches]
            self.assertTrue(torch.equal(inputs["input"], ref_inputs["input"]))
            self.assertTrue(torch.equal(inputs["positions"], ref_inputs["positions"]))
            self.assertTrue(torch.equal(labels, ref_labels))

        # The cached batches themselves must still be distinct data.
        self.assertFalse(torch.equal(batches[0][1], batches[1][1]))

    def test_source_is_read_only_once(self):
        num_samples = 4
        source = DummyDataset()
        dataset = CachedBatchDataset(source, num_samples)

        data_iter = iter(dataset)
        samples = [next(data_iter) for _ in range(3 * num_samples)]

        # DummyDataset yields strictly increasing values, so a second read of the
        # source would show values beyond the cached prefix.
        for i, (sample, label) in enumerate(samples):
            self.assertEqual(sample["input"], i % num_samples)
            self.assertEqual(label, i % num_samples)

    def test_raises_when_source_is_too_short(self):
        dataset = CachedBatchDataset(DummyDataset(), num_samples=101)
        with self.assertRaisesRegex(ValueError, "ran out after 100 samples"):
            next(iter(dataset))

    def test_rejects_non_positive_num_samples(self):
        with self.assertRaises(ValueError):
            CachedBatchDataset(DummyDataset(), num_samples=0)

    def test_state_dict_resumes_the_cycle(self):
        num_samples = 4
        dataset = CachedBatchDataset(DummyDataset(), num_samples)

        data_iter = iter(dataset)
        for _ in range(num_samples + 1):
            next(data_iter)
        state_dict = dataset.state_dict()

        resumed = CachedBatchDataset(DummyDataset(), num_samples)
        resumed.load_state_dict(state_dict)
        sample, label = next(iter(resumed))

        self.assertEqual(sample["input"], 1)
        self.assertEqual(label, 1)

    def test_interleaved_rejects_num_cached_batches(self):
        with self.assertRaisesRegex(ValueError, "num_cached_batches"):
            InterleavedHuggingFaceTextDataLoader.Config(
                sources=[HFDataSource(dataset="c4_test", num_cached_batches=1)]
            )


if __name__ == "__main__":
    unittest.main()
