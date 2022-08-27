# type: ignore

#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
#

import collections
import inspect
import math
import os
import random
import re
import sys
import time
import warnings
from collections.abc import Sized
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

from packaging import version
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import Trainer as BaseTrainer
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.utils import (
    is_apex_available,
)

if is_apex_available():
    from apex import amp
import torch.distributed as dist
from torch import nn
from transformers.deepspeed import deepspeed_init, deepspeed_reinit
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    get_last_checkpoint,
    has_length,
    set_seed,
    speed_metrics,
)

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
from transformers.integrations import (  # isort: split
    hp_params,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalLoopOutput, SchedulerType
from transformers.utils import logging

from merlin_standard_lib import Schema

from ..config.trainer import T4RecTrainingArguments
from .model.base import Model
from .utils.data_utils import T4RecDataLoader
import pandas as pd
from vvrecsys.datasets.reader import Reader

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
top_k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
random_items = torch.rand((512, 2911))
random_items = random_items.cuda()

dataset_info = Reader('bigdata.h5')
items_encoder = dataset_info.items_encoder

df = pd.read_csv('poptop_item.csv')
top_items = df.id_tov_cl.to_list()
top_items_tns = torch.zeros(len(top_items) + 3)

top_items = items_encoder.transform(top_items)
value = 0.99
for item in top_items:
    top_items_tns[item + 1] = value
    value -= 0.0001
top_k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

top_items_tns = top_items_tns.unsqueeze(0).repeat(512, 1).cuda()


class Trainer(BaseTrainer):
    """
    An :class:`~transformers.Trainer` specialized for sequential recommendation
    including (session-based and sequtial recommendation)

    Parameters
    ----------
    model: Model
        The Model defined using Transformers4Rec api.
    args: T4RecTrainingArguments
        The training arguments needed to setup training and evaluation
        experiments.
    schema: Optional[Dataset.schema], optional
        The schema object including features to use and their properties.
        by default None
    train_dataset_or_path: Optional[Union[str, Dataset]], optional
        Path of parquet files or DataSet to use for training.
        by default None
    eval_dataset_or_path: Optional[str, Dataset], optional
        Path of parquet files or DataSet to use for evaluation.
        by default None
    train_dataloader: Optional[DataLoader], optional
        The data generator to use for training.
        by default None
    eval_dataloader: Optional[DataLoader], optional
        The data generator to use for evaluation.
        by default None
    compute_metrics: Optional[bool], optional
        Whether to compute metrics defined by Model class or not.
        by default None
    incremental_logging: bool
        Whether to enable incremental logging or not. If True, it ensures that
        global steps are incremented over many `trainer.train()` calls, so that
        train and eval metrics steps do not overlap and can be seen properly
        in reports like W&B and Tensorboard
    """

    def __init__(
            self,
            model: Model,
            args: T4RecTrainingArguments,
            schema: Schema = None,
            train_dataset_or_path=None,
            eval_dataset_or_path=None,
            test_dataset_or_path=None,
            train_dataloader: Optional[DataLoader] = None,
            eval_dataloader: Optional[DataLoader] = None,
            test_dataloader: Optional[DataLoader] = None,
            callbacks: Optional[List[TrainerCallback]] = [],
            compute_metrics=None,
            incremental_logging: bool = False,
            max_steps_eval: int = None,
            clean_out: bool = False,
            **kwargs,
    ):

        mock_dataset = DatasetMock()
        hf_model = HFWrapper(model)

        self.incremental_logging = incremental_logging
        if self.incremental_logging:
            self.past_global_steps = 0
            incremental_logging_callback = IncrementalLoggingCallback(self)
            callbacks.append(incremental_logging_callback)

        super(Trainer, self).__init__(
            model=hf_model,
            args=args,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            callbacks=callbacks,
            **kwargs,
        )

        self.compute_metrics = compute_metrics
        self.train_dataset_or_path = train_dataset_or_path
        self.eval_dataset_or_path = eval_dataset_or_path
        self.test_dataset_or_path = test_dataset_or_path
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.schema = schema
        self.incremental_logging = incremental_logging
        self.max_steps_eval = max_steps_eval
        self.clean_out = clean_out

    def get_train_dataloader(self):
        """
        Set the train dataloader to use by Trainer.
        It supports user defined data-loader set as an attribute in the constructor.
        When the attribute is None, The data-loader is defined using train_dataset
        and the `data_loader_engine` specified in Training Arguments.
        """
        if self.train_dataloader is not None:
            return self.train_dataloader

        assert self.schema is not None, "schema is required to generate Train Dataloader"
        return T4RecDataLoader.parse(self.args.data_loader_engine).from_schema(
            self.schema,
            self.train_dataset_or_path,
            self.args.per_device_train_batch_size,
            max_sequence_length=self.args.max_sequence_length,
            drop_last=self.args.dataloader_drop_last,
            shuffle=True,
            shuffle_buffer_size=self.args.shuffle_buffer_size,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Set the eval dataloader to use by Trainer.
        It supports user defined data-loader set as an attribute in the constructor.
        When the attribute is None, The data-loader is defined using eval_dataset
        and the `data_loader_engine` specified in Training Arguments.
        """
        if self.eval_dataloader is not None:
            return self.eval_dataloader

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert self.schema is not None, "schema is required to generate Eval Dataloader"
        return T4RecDataLoader.parse(self.args.data_loader_engine).from_schema(
            self.schema,
            self.eval_dataset_or_path,
            self.args.per_device_eval_batch_size,
            max_sequence_length=self.args.max_sequence_length,
            drop_last=self.args.dataloader_drop_last,
            shuffle=False,
            shuffle_buffer_size=self.args.shuffle_buffer_size,
        )

    def get_test_dataloader(self, test_dataset=None):
        """
        Set the test dataloader to use by Trainer.
        It supports user defined data-loader set as an attribute in the constructor.
        When the attribute is None, The data-loader is defined using test_dataset
        and the `data_loader_engine` specified in Training Arguments.
        """
        if self.test_dataloader is not None:
            return self.test_dataloader

        if test_dataset is None and self.test_dataset is None:
            raise ValueError("Trainer: test requires an test_dataset.")
        test_dataset = test_dataset if test_dataset is not None else self.test_dataset
        assert self.schema is not None, "schema is required to generate Test Dataloader"
        return T4RecDataLoader.parse(self.args.data_loader_engine).from_schema(
            self.schema,
            self.test_dataset_or_path,
            self.args.per_device_eval_batch_size,
            max_sequence_length=self.args.max_sequence_length,
            drop_last=self.args.dataloader_drop_last,
            shuffle=False,
            shuffle_buffer_size=self.args.shuffle_buffer_size,
        )

    def num_examples(self, dataloader: DataLoader):
        """
        Overriding :obj:`Trainer.num_examples()` method because
        the data loaders for this project do not return the dataset size,
        but the number of steps. So we estimate the dataset size here
        by multiplying the number of steps * batch size
        """
        """
        if dataloader == self.get_train_dataloader():
            batch_size = self.args.per_device_train_batch_size
        else:
            batch_size = self.args.per_device_eval_batch_size
        """
        return len(dataloader) * dataloader._batch_size

    def reset_lr_scheduler(self) -> None:
        """
        Resets the LR scheduler of the previous :obj:`Trainer.train()` call,
        so that a new LR scheduler one is created by the next :obj:`Trainer.train()` call.
        This is important for LR schedules like `get_linear_schedule_with_warmup()`
        which decays LR to 0 in the end of the train
        """
        self.lr_scheduler = None

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        # flexibility in scheduler with num_cycles as hyperparams
        if self.lr_scheduler is None:
            self.lr_scheduler = self.get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.args.learning_rate_num_cosine_cycles_by_epoch
                           * self.args.num_train_epochs,
            )

    # Override the method get_scheduler to accept num_cycle params ?
    # The advantage is to use the unified HF API with many scheduler
    # we can also send a PR to HF ?
    @staticmethod
    def get_scheduler(
            name: Union[str, SchedulerType],
            optimizer: Optimizer,
            num_warmup_steps: Optional[int] = None,
            num_training_steps: Optional[int] = None,
            num_cycles: Optional[int] = 0.5,
    ):
        """
        Unified API to get any scheduler from its name.

        Parameters
        ----------
        name: (:obj:`str` or `:obj:`SchedulerType`)
            The name of the scheduler to use.
        optimizer: (:obj:`torch.optim.Optimizer`)
            The optimizer that will be used during training.
        num_warmup_steps: (:obj:`int`, `optional`)
            The number of warmup steps to do. This is not required by all schedulers
            (hence the argument being optional),
            the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps: (:obj:`int`, `optional`)
            The number of training steps to do. This is not required by all schedulers
            (hence the argument being optional),
            the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles: (:obj:`int`, `optional`)
            The number of waves in the cosine schedule /
            hard restarts to use for cosine scheduler
        """
        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
        if name == SchedulerType.CONSTANT:
            return schedule_func(optimizer)

        # All other schedulers require `num_warmup_steps`
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

        # All other schedulers require `num_training_steps`
        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

        if "num_cycles" in inspect.signature(schedule_func).parameters:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            )

        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

    def prediction_step(
            self,
            model: torch.nn.Module,
            inputs: Dict[str, torch.Tensor],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
            ignore_masking: bool = False,
    ) -> Tuple[
        Optional[float],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, Any]],
    ]:
        """
        Overriding :obj:`Trainer.prediction_step()`
        to provide more flexibility to unpack results from the model,
        like returning labels that are not exactly one input feature
        model
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(inputs, training=False, ignore_masking=ignore_masking)
            else:
                outputs = model(inputs, training=False, ignore_masking=ignore_masking)

            loss = outputs["loss"].mean().detach()

        if prediction_loss_only:
            return (loss, None, None, None)

        predictions = outputs["predictions"].detach()
        labels = outputs["labels"].detach()

        # TODO: define metadata dict in the model for logging
        # other_outputs = {
        #    k: v.detach() if isinstance(v, torch.Tensor) else v
        #    for k, v in outputs.items()
        #    if k not in ignore_keys + ["loss", "predictions", "labels"]
        # }
        other_outputs = None

        return (loss, predictions, labels, other_outputs)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: Optional[str] = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding :obj:`Trainer.prediction_loop()`
        (shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`)
        to provide more flexibility to work with streaming metrics
        (computed at each eval batch) and
        to log with the outputs of the model
        (e.g. prediction scores, prediction metadata, attention weights)

        Parameters
        ----------
        dataloader: DataLoader
            DataLoader object to use to iterate over evaluation data
        description: str
            Parameter to describe the evaluation experiment.
            e.g: `Prediction`, `test`
        prediction_loss_only: Optional[bool]
            Whether or not to return the loss only.
            by default None
        ignore_keys: Optional[List[str]]
            Columns not accepted by the ``model.forward()`` method
            are automatically removed.
            by default None
        metric_key_prefix: Optional[str]
            Prefix to use when logging evaluation metrics.
            by default `eval`
        """
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        if description == "Prediction":
            ignore_masking = True
        else:
            ignore_masking = False

        # set the model
        model = self.model.module
        # reset metrics for the dataset (Train, Valid or Test)
        if self.compute_metrics:
            model.reset_metrics()

        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")

        batch_size = dataloader._batch_size

        logger.info("***** Running %s *****", description)
        logger.info("  Batch size = %d", batch_size)

        preds_item_ids_scores_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        if metric_key_prefix == "train" and self.args.eval_steps_on_train_set:
            num_examples = self.args.eval_steps_on_train_set * batch_size
        else:
            num_examples = self.num_examples(dataloader)

        logger.info("  Num sessions (examples) = %d", num_examples)

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_item_ids_scores_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds_item_ids_scores = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        train_loader = iter(self.train_dataloader)
        val_loader = iter(self.eval_dataloader)
        predict_metrics = []
        cleanout_metrics = []
        random_metrics = []
        poptop_metrics = []
        # Iterate over dataloader
        for step in range(len(self.eval_dataloader)):
            if step == self.max_steps_eval:
                break
            labels = next(val_loader)
            labels = labels['item_id-list_trim'].cuda()
            train_inputs = next(train_loader)

            # Update the observed num examples
            observed_batch_size = find_batch_size(train_inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Limits the number of evaluation steps on train set (which is usually larger)
            if (
                    metric_key_prefix == "train"
                    and self.args.eval_steps_on_train_set > 0
                    and step + 1 > self.args.eval_steps_on_train_set
            ):
                break

            loss, preds, _, outputs = self.prediction_step(
                model,
                train_inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
                ignore_masking=ignore_masking,
            )

            # Updates metrics
            # TODO: compute metrics each N eval_steps to speedup evaluation
            metrics_results_detailed = None
            if self.compute_metrics:
                if step % self.args.compute_metrics_each_n_steps == 0:
                    metrics_results_detailed = model.calculate_metrics(
                        preds, labels, mode=metric_key_prefix, forward=False, call_body=False
                    )
                    predict_metrics.append(metrics_results_detailed)
                    random_metrics.append(
                        model.calculate_metrics(
                            random_items, labels, mode=metric_key_prefix, forward=False, call_body=False
                        )
                    )
                    preds_clean_out = preds.clone()
                    preds_clean_out = preds_clean_out.scatter_(1, train_inputs['item_id-list_trim'].cuda(), -100)
                    cleanout_metrics.append(
                        model.calculate_metrics(
                            preds_clean_out, labels, mode=metric_key_prefix, forward=False, call_body=False
                        )
                    )
                    poptop_metrics.append(
                        model.calculate_metrics(
                            top_items_tns, labels, mode=metric_key_prefix, forward=False, call_body=False
                        )
                    )

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = (
                    losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
                )
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=0)
                )
            if preds is not None and self.args.predict_top_k > 0:
                preds_sorted_item_scores, preds_sorted_item_ids = torch.topk(
                    preds, k=self.args.predict_top_k, dim=-1
                )
                self._maybe_log_predictions(
                    labels,
                    preds_sorted_item_ids,
                    preds_sorted_item_scores,
                    # outputs["pred_metadata"],
                    metrics_results_detailed,
                    metric_key_prefix,
                )
                # The output predictions will be a tuple with the ranked top-n item ids,
                # and item recommendation scores
                preds_item_ids_scores = (
                    preds_sorted_item_ids,
                    preds_sorted_item_scores,
                )
                preds_item_ids_scores_host = (
                    preds_item_ids_scores
                    if preds_item_ids_scores_host is None
                    else nested_concat(
                        preds_item_ids_scores_host,
                        preds_item_ids_scores,
                    )
                )

            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU
            # if we have done enough accumulation steps.
            if (
                    self.args.eval_accumulation_steps is not None
                    and (step + 1) % self.args.eval_accumulation_steps == 0
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(all_labels, labels, padding_index=0)
                    )
                if preds_item_ids_scores_host is not None:
                    preds_item_ids_scores = nested_numpify(preds_item_ids_scores_host)
                    all_preds_item_ids_scores = (
                        preds_item_ids_scores
                        if all_preds_item_ids_scores is None
                        else nested_concat(
                            all_preds_item_ids_scores,
                            preds_item_ids_scores,
                        )
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_item_ids_scores_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels if all_labels is None else nested_concat(all_labels, labels, padding_index=0)
            )
        if preds_item_ids_scores_host is not None:
            preds_item_ids_scores = nested_numpify(preds_item_ids_scores_host)
            all_preds_item_ids_scores = (
                preds_item_ids_scores
                if all_preds_item_ids_scores is None
                else nested_concat(
                    all_preds_item_ids_scores,
                    preds_item_ids_scores,
                )
            )
        # Get Number of samples :
        # the data loaders for this project do not return the dataset size,
        num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size
        # and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds_item_ids_scores is not None:
            all_preds_item_ids_scores = nested_truncate(all_preds_item_ids_scores, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Get metrics :
        metrics = {}
        # Computing the metrics results as the average of all steps
        if self.compute_metrics:
            streaming_metrics_results = model.compute_metrics(mode=metric_key_prefix)
            streaming_metrics_results_flattened = process_metrics(
                streaming_metrics_results, prefix=metric_key_prefix + "_/"
            )

            metrics = {**metrics, **streaming_metrics_results_flattened}

        metrics[f"{metric_key_prefix}_/loss"] = all_losses.mean().item()

        predict_metrics = self.metric_count(predict_metrics)
        cleanout_metrics = self.metric_count(cleanout_metrics)
        random_metrics = self.metric_count(random_metrics)
        poptop_metrics = self.metric_count(poptop_metrics)

        print('\nrandom')
        for key in sorted(random_metrics.keys()):
            print(" %s = %s" % (key, str([round(i, 4) for i in random_metrics[key].tolist()])))

        print('\npoptop')
        for key in sorted(poptop_metrics.keys()):
            print(" %s = %s" % (key, str([round(i, 4) for i in poptop_metrics[key].tolist()])))

        print('\npredict')
        for key in sorted(predict_metrics.keys()):
            print(" %s = %s" % (key, str([round(i, 4) for i in predict_metrics[key].tolist()])))

        print('\ncleanout')
        for key in sorted(cleanout_metrics.keys()):
            print(" %s = %s" % (key, str([round(i, 4) for i in cleanout_metrics[key].tolist()])))

        self.plot_metrics(predict_metrics, cleanout_metrics, random_metrics, poptop_metrics,
                          name=['predict', 'cleanout', 'random', 'poptop'])

        return EvalLoopOutput(
            predictions=all_preds_item_ids_scores,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_examples,
        )

    def metric_count(self, values_metrics):
        ndcg = torch.mean(torch.concat([i['next-item/ndcg_at'].unsqueeze(0) for i in values_metrics]), dim=0)
        avg_precision = torch.mean(torch.concat([i['next-item/precision_at'].unsqueeze(0) for i in values_metrics]),
                                   dim=0)
        recall = torch.mean(torch.concat([i['next-item/recall_at'].unsqueeze(0) for i in values_metrics]), dim=0)
        map = torch.mean(torch.concat([i['next-item/mean_recipricol_rank_at'].unsqueeze(0) for i in values_metrics]),
                         dim=0)
        return {
            'next-item/ndcg_at': ndcg,
            'next-item/avg_precision_at': avg_precision,
            'next-item/recall_at': recall,
            'next-item/mean_recipricol_rank_at': map,
        }

    def plot_metrics(self, predict_metrics, cleanout_metrics, random_metrics, poptop_metrics, name):
        for *metric, name_metrics in zip(predict_metrics.values(), cleanout_metrics.values(), random_metrics.values(),
                                         poptop_metrics.values(), predict_metrics.keys()):
            [plt.plot(top_k, i.cpu().detach().numpy(), 'o-', label=j) for i, j in zip(metric, name)]

            plt.xlabel(f'x - top k {name_metrics.split("/")[1]}')
            plt.ylabel('y - score')
            plt.legend()
            plt.savefig(f'{name_metrics.split("/")[1]}.png')
            plt.show()

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warning(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

                # release memory
                del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE or is_sagemaker_mp_enabled()
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                if self.deepspeed:
                    # temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
                    deepspeed_engine, optimizer, lr_scheduler = deepspeed_reinit(self)
                    self.model = deepspeed_engine.module
                    self.model_wrapped = deepspeed_engine
                    self.deepspeed = deepspeed_engine
                    self.optimizer = optimizer
                    self.lr_scheduler = lr_scheduler
                    self.deepspeed.load_checkpoint(
                        self.state.best_model_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
                    )
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = torch.load(best_model_path, map_location="cpu")
                    # If the model is on the GPU, it still works!
                    self._load_state_dict_in_model(state_dict)
            else:
                logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_model_and_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save_model_and_checkpoint(self, model, trial, metrics, save_model_class=True):
        """
        Save the serialized model + trainer and random states.

        Parameters
        ----------
        save_model_class: Optional[bool]
            Whether to save the Model class or not.
            by default False
        """
        try:
            import cloudpickle
        except ImportError:
            cloudpickle = None

        logger.info("Saving model...")
        output_dir = os.path.join(
            self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        )

        # save model parameters
        self._save_checkpoint(model, trial, metrics=metrics)
        # save the serialized model
        if save_model_class:
            # TODO : fix serialization of DatasetSchema object
            if cloudpickle is None:
                raise ValueError("cloudpickle is required to save model class")

            with open(os.path.join(output_dir, "model_class.pkl"), "wb") as out:
                cloudpickle.dump(self.model.module, out)

    def load_model_trainer_states_from_checkpoint(self, checkpoint_path, model=None):
        """
        This method loads the checkpoints states of the model, trainer and random states.
        If model is None the serialized model class is loaded from checkpoint.
        It does not loads the optimizer and LR scheduler states (for that call trainer.train()
        with resume_from_checkpoint argument for a complete load)

        Parameters
        ----------
        checkpoint_path: str
            Path to the checkpoint directory.
        model: Optional[Model]
            Model class used by Trainer. by default None
        """
        import os

        if model is None:
            try:
                import cloudpickle
            except ImportError:
                raise ImportError("cloudpickle is required to load model class")
            logger.info("Loading model class")
            model = cloudpickle.load(open(os.path.join(checkpoint_path, "model_class.pkl"), "rb"))

        self.model = HFWrapper(model)
        logger.info("Loading weights of previously trained model")
        # Restoring model weights
        self.model.load_state_dict(
            # torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin"))
            torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
        )
        # Restoring random state
        rng_file = os.path.join(checkpoint_path, "rng_state.pth")
        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
        # Restoring AMP scaler
        if self.use_amp:
            self.scaler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scaler.pt")))

    @property
    def log_predictions_callback(self) -> Callable:
        return self.__log_predictions_callback

    @log_predictions_callback.setter
    def log_predictions_callback(self, var: Callable):
        self.__log_predictions_callback = var

    def _maybe_log_predictions(
            self,
            labels: torch.Tensor,
            pred_item_ids: torch.Tensor,
            pred_item_scores: torch.Tensor,
            metrics: Dict[str, np.ndarray],
            metric_key_prefix: str,
    ):
        """
        If --log_predictions is enabled, calls a callback function to
        log predicted item ids, scores, metadata and metrics.
        Parameters
        ----------
        labels: torch.Tensor
            True labels.
        pred_item_ids: torch.Tensor
            The predicted items ids. if top_k is set:
            we return to top-k items for each
            next-item prediction.
        pred_item_scores: torch.Tensor
            The prediction scores, if top_k is set:
            we return to top-k predictions for each
            next-item prediction.
        metrics: Dict[str, np.ndarray]
            Dictionary of metrics computed by Model.
        metric_key_prefix: str
            Prefix to use when logging evaluation metrics.
            by default `eval`
        """
        # TODO Add pred_metadata: Dict[str, torch.Tensor],

        if self.args.log_predictions and self.log_predictions_callback is not None:
            # Converting torch Tensors to NumPy and callback predictions logging function
            # preds_metadata = {k: v.cpu().numpy() for k, v in pred_metadata.items()}

            self.log_predictions_callback(
                labels=labels.cpu().numpy(),
                pred_item_ids=pred_item_ids.cpu().numpy(),
                pred_item_scores=pred_item_scores.cpu()
                    .numpy()
                    .astype(np.float32),  # Because it is float16 when --fp16
                # preds_metadata=preds_metadata,
                metrics=metrics,
                dataset_type=metric_key_prefix,
            )

    def _increment_past_global_steps(self, current_global_step: int):
        self.past_global_steps += current_global_step

    def _get_general_global_step(self) -> int:
        general_global_step = self.past_global_steps
        if self.model.training:
            general_global_step += self.state.global_step

        return general_global_step

    def log(self, logs: Dict[str, float]) -> None:
        # Ensuring that eval metrics are prefixed as "eval_" so that the HF integration loggers
        # do not prefix metrics names with 'train/' (as 'train/' is always added when not eval)
        logs = {re.sub("^eval/", "eval_", k).replace("train/", ""): v for k, v in logs.items()}

        if not self.incremental_logging:
            super().log(logs)
        else:
            # If Incremental logging is enabled, ensures that global steps are always
            # incremented after train() calls
            # so that metrics are logger with no overlap on W&B and Tensorboard
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)

            # As state.global_step is also used for the learning rate schedules,
            # we create a copy only for logging
            state_copy = deepcopy(self.state)
            state_copy.global_step = self._get_general_global_step()

            output = {**logs, **{"step": state_copy.global_step}}
            self.state.log_history.append(output)
            self.control = self.callback_handler.on_log(self.args, state_copy, self.control, logs)


def process_metrics(metrics, prefix="", to_cpu=True):
    metrics_proc = {}
    for root_key, root_value in metrics.items():
        if isinstance(root_value, dict):
            flattened_metrics = process_metrics(root_value, prefix=prefix, to_cpu=to_cpu)
            metrics_proc = {**metrics_proc, **flattened_metrics}
        else:
            value = root_value.cpu().numpy().item() if to_cpu else root_value
            metrics_proc[f"{prefix}{root_key}"] = value
    return metrics_proc


class IncrementalLoggingCallback(TrainerCallback):
    """
    An :class:`~transformers.TrainerCallback` that changes the state of the Trainer
    on specific hooks for the purpose of the incremental logging
    Parameters
    ----------
    trainer: Trainer
    """

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        pass

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Increments the global steps for logging with the global steps of the last train()
        self.trainer._increment_past_global_steps(state.global_step)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Evaluates on eval set
        # self.trainer.evaluate()
        pass


class DatasetMock(Dataset, Sized):
    """
    Mock to inform HF Trainer that the dataset is sized,
    and can be obtained via the generated/provided data loader
    """

    def __init__(self, nsteps=1):
        self.nsteps = nsteps

    def __len__(self):
        return self.nsteps


class HFWrapper(torch.nn.Module):
    """
    Prepare the signature of the forward method
    as required by HF Trainer
    """

    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, *args, **kwargs):
        inputs = kwargs
        return self.module(inputs, *args)
