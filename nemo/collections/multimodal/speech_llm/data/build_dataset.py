# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import copy
from pathlib import Path

import torch
from megatron.core import parallel_state
from omegaconf.omegaconf import OmegaConf

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextLhotseDataset,
    get_prompt_format_fn,
)
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.multimodal.speech_llm.data.audio_text_dataset import (
    get_audio_text_dataset_from_config,
    get_tarred_audio_text_dataset_from_config,
)
from nemo.collections.multimodal.speech_llm.data.lhotse_dataset import LhotseAudioQuestionAnswerDataset, TextProcessing
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.utils import logging


def build_salm_dataset(model_instance, data_cfg, is_train):
    if 'augmentor' in data_cfg:
        augmentor = process_augmentations(
            data_cfg['augmentor'], global_rank=model_instance.global_rank, world_size=model_instance.world_size
        )
    else:
        augmentor = None

    # Check dataset max_seq_legnth and max_position_embeddings size
    if (
        model_instance.cfg.get('position_embedding_type', None) in [None, 'learned_absolute']
        and data_cfg.max_seq_length > model_instance.cfg.max_position_embeddings
    ):
        logging.warning(
            f"Set dataset max_seq_length to max_position_embeddings {model_instance.cfg.max_position_embeddings} if using learned_absolute position embedding"
        )
        data_cfg.max_seq_length = model_instance.cfg.max_position_embeddings

    # Notably, the data weights are controlled by either bucketing_weights
    # or concat_sampling_probabilities depending on the dataset type.
    if data_cfg.get("use_lhotse"):
        tp = TextProcessing(
            model_instance.tokenizer,
            max_seq_length=data_cfg["max_seq_length"],
            min_seq_length=data_cfg["min_seq_length"],
            add_bos=data_cfg.get('add_bos', False),
            add_eos=data_cfg.get('add_eos', False),
            add_sep=data_cfg.get('add_sep', False),
            sep_id=model_instance.sep_id,
            seed=data_cfg.get('seed', 1234),
            separate_prompt_and_response_with_newline=data_cfg.get('separate_prompt_and_response_with_newline', True),
            answer_only_loss=model_instance.cfg.get('answer_only_loss', True),
            truncation_field=data_cfg.get('truncation_field', 'context'),
            pad_to_max_length=data_cfg.get('pad_to_max_length', False),
            prompt_template=data_cfg.get('prompt_template', None),
            virtual_tokens=model_instance.virtual_tokens,
            tokens_to_generate=data_cfg.get(
                'tokens_to_generate', 0
            ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
            input_key=data_cfg.get('input_key', 'input'),
            output_key=data_cfg.get('output_key', 'output'),
            end_string=data_cfg.get('end_string', None),
            sample_alpha=data_cfg.get('sample_alpha', None),
            input_text_mask_ratio=data_cfg.get('input_text_mask_ratio', None),
        )
        if model_instance.cfg.perception.get("is_canary", False):
            prompt_tokenizer = (
                model_instance.perception.tokenizer
                if hasattr(model_instance.perception, "tokenizer")
                else model_instance.perception.asr_model.tokenizer
            )
            prompt_tokenizer = TokenizerWrapper(prompt_tokenizer)
            prompt_format_fn = get_prompt_format_fn('canary')
        else:
            prompt_format_fn = None
            prompt_tokenizer = None
        return LhotseAudioQuestionAnswerDataset(
            tp,
            default_question="answer the question according to the previous audio",
            tokens_to_generate=data_cfg.get('tokens_to_generate', 0),
            pad_to_max_length=data_cfg.get('pad_to_max_length', False),
            max_seq_length=data_cfg["max_seq_length"],
            prompt_format_fn=prompt_format_fn,
            prompt_tokenizer=prompt_tokenizer,
            convert_canary_prompt_to_text=data_cfg.get('convert_canary_prompt_to_text', False),
            prepend_to_exist_question=data_cfg.get('prepend_to_exist_question', None),
            random_context_prob=data_cfg.get('random_context_prob', 0.0),
            random_context_positive_percent=data_cfg.get('random_context_positive_percent', 0.1),
        )

    # Notably, the data weights are controlled by either bucketing_weights
    # or concat_sampling_probabilities depending on the dataset type.
    if data_cfg.get('is_tarred', False):
        return get_tarred_audio_text_dataset_from_config(
            config=data_cfg,
            tokenizer=model_instance.tokenizer,
            augmentor=augmentor,
            sep_id=model_instance.sep_id,
            answer_only_loss=model_instance.cfg.get('answer_only_loss', True),
            virtual_tokens=model_instance.virtual_tokens,
            global_rank=parallel_state.get_data_parallel_rank(),
            world_size=parallel_state.get_data_parallel_world_size(),
        )
    else:
        return get_audio_text_dataset_from_config(
            manifest_filepath=data_cfg.manifest_filepath,
            config=data_cfg,
            tokenizer=model_instance.tokenizer,
            augmentor=augmentor,
            is_train=is_train,
            sep_id=model_instance.sep_id,
            answer_only_loss=model_instance.cfg.get('answer_only_loss', True),
            virtual_tokens=model_instance.virtual_tokens,
        )


def build_salm_dataloader(dataset, data_cfg, consumed_samples=0, is_predict=False, is_eval=False):
    """Buld dataloader given an input dataset."""
    if data_cfg.get("use_lhotse"):
        if is_eval == False and is_predict == False:
            return get_lhotse_dataloader_from_config(
                data_cfg,
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
                dataset=dataset,
            )
        # for eval, we need to create separate dataset so as to report splitted numbers
        else:
            dls = []
            for dataset_idx, (cur_manifest_filepath) in enumerate(data_cfg.manifest_filepath):
                conf = copy.deepcopy(data_cfg)
                conf['manifest_filepath'] = cur_manifest_filepath
                dls.append(
                    get_lhotse_dataloader_from_config(
                        conf,
                        global_rank=parallel_state.get_data_parallel_rank(),
                        world_size=parallel_state.get_data_parallel_world_size(),
                        dataset=dataset,
                    )
                )
            if 'names' not in data_cfg:
                names = []
                for cur_manifest_filepath in data_cfg.manifest_filepath:
                    names.append(Path(cur_manifest_filepath).stem)
                OmegaConf.update(data_cfg, 'names', names, force_add=True)
                logging.info(f'Update dataset names as {names}')
            return dls

    logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
    if isinstance(dataset, BlendableDataset):
        collate_fn = dataset.datasets[0].collate_fn
    elif hasattr(dataset, 'collate_fn'):
        collate_fn = dataset.collate_fn
    elif hasattr(dataset.datasets[0], 'collate_fn'):
        # support datasets that are lists of entries
        collate_fn = dataset.datasets[0].collate_fn
    else:
        # support datasets that are lists of lists
        collate_fn = dataset.datasets[0].datasets[0].collate_fn

    if isinstance(dataset, torch.utils.data.IterableDataset):
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        num_micro_batches = data_cfg.global_batch_size // (data_cfg.micro_batch_size * data_parallel_size)
        global_batch_size_on_this_data_parallel_rank = num_micro_batches * data_cfg.micro_batch_size

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            shuffle=False,
            batch_size=global_batch_size_on_this_data_parallel_rank,
            drop_last=True,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
        )
        return dataloader

    if is_predict:
        # MegatronPretrainingBatchSampler doesn't work with trainer.predict()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=data_cfg.micro_batch_size,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
        )
        return dataloader

    batch_sampler = MegatronPretrainingBatchSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=data_cfg.micro_batch_size,
        global_batch_size=data_cfg.global_batch_size,
        data_parallel_rank=parallel_state.get_data_parallel_rank(),
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
        drop_last=data_cfg.drop_last,
        pad_samples_to_global_batch_size=not data_cfg.drop_last,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=True if data_cfg.num_workers > 0 else False,
    )
    return dataloader
