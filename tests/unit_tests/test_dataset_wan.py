# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from datasets import load_dataset

from torchtitan.config import ConfigManager
from torchtitan.hf_datasets import DatasetConfig


class TestWanDataLoader(unittest.TestCase):
    def setUp(self):
    
        # Import here to avoid circular import during test collection
        from torchtitan.experiments.wan.wan_datasets import (
            DATASETS,
            _load_raw_video_dataset,
            _process_raw_video_sample   
        )

        # Store reference for use in tearDown
        self._DATASETS = DATASETS
        self._1xwm_data_processor = _process_raw_video_sample

        self._DATASETS["1xwm"] = DatasetConfig(
            path="tests/assets/1xwm_test",
            loader=_load_raw_video_dataset,
            sample_processor=self._1xwm_data_processor,
        )

    def tearDown(self):
        del self._DATASETS["1xwm"]

    def test_load_dataset(self):
        from torchtitan.experiments.wan.wan_datasets import build_wan_dataloader

        # The test checks for the correct tensor shapes during the first num_steps
        # The next num_steps ensure the loaded from checkpoint dataloader generates tokens and labels correctly
        for world_size in [2]:
            for rank in range(world_size):
                dataset_name = "1xwm"
                batch_size = 1
                num_frames = 21
                H, W, C = 512, 512, 3

                num_steps = 15

                # TODO: if num_steps * batch_size * world_size is larger than the number of samples
                # in the dataset, then the test will fail, due to huggingface's
                # non-resumption when checkpointing after the first epoch

                path = "torchtitan.experiments.wan.job_config"
                config_manager = ConfigManager()
                config = config_manager.parse_args(
                    [
                        f"--job.custom_config_module={path}",
                        "--training.dataset",
                        dataset_name,
                        "--training.dataset_path",
                        "tests/assets/1xwm_test",
                        "--training.local_batch_size",
                        str(batch_size),
                        "--training.downsampled",
                        "4",
                        "--training.clip_length",
                        "77",
                        "--training.window_size",
                        "8",
                        "--training.robot_temporal_mode",
                        "downsample",
                        "--training.classifier_free_guidance_prob",
                        "0.447",
                        "--training.test_mode",
                        "--encoder.t5_encoder",
                        "tests/assets/flux_test_encoders/t5-v1_1-xxl",
                        "--encoder.clip_encoder",
                        "tests/assets/flux_test_encoders/clip-vit-large-patch14",
                    ]
                )

                dl = build_wan_dataloader(
                    dp_world_size=world_size,
                    dp_rank=rank,
                    job_config=config,
                    tokenizer=None,
                    infinite=True,
                )

                it = iter(dl)

                for i in range(0, num_steps):
                    input_data, videos = next(it)
                    
                    # Extract robot_states from input_data
                    states = input_data["robot_states"]

                    assert (
                        len(input_data) == 5
                    )  # (clip_tokens, t5_tokens, prompt, robot_states, video_frames)
                    # TODO: update this to be just t5_encodings and allow to have them to be fixed to "" encoding
                    assert videos.shape == (batch_size, num_frames, H, W, C)
                    assert states.shape == (batch_size, num_frames, 25)
                    assert input_data["clip_tokens"].shape == (
                        batch_size,
                        77,
                    )
                    assert input_data["t5_tokens"].shape == (
                        batch_size,
                        256,
                    )

                state = dl.state_dict()

                # Create new dataloader, restore checkpoint, and check if next data yielded is the same as above
                dl_resumed = build_wan_dataloader(
                    dp_world_size=world_size,
                    dp_rank=rank,
                    job_config=config,
                    tokenizer=None,
                    infinite=True,
                )
                dl_resumed.load_state_dict(state)
                it_resumed = iter(dl_resumed)

                # After checkpointing, the resumed dataloader should produce the same samples
                # starting from where we checkpointed. Both iterators should be at the same position.
                for i in range(num_steps):
                    # Set torch manual seed before each dataloader iteration to ensure consistent randomness
                    # across dataloaders for testing purposes.
                    torch.manual_seed(i)
                    expected_input_ids, expected_videos = next(it)
                    expected_states = expected_input_ids["robot_states"]
                    torch.manual_seed(i)
                    input_ids, videos = next(it_resumed)
                    states = input_ids["robot_states"]

                    assert torch.equal(
                        input_ids["clip_tokens"], expected_input_ids["clip_tokens"]
                    )
                    assert torch.equal(
                        input_ids["t5_tokens"], expected_input_ids["t5_tokens"]
                    )
                    assert torch.equal(videos, expected_videos)
                    assert torch.equal(states, expected_states)
