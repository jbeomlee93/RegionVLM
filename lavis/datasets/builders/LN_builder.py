"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset
from lavis.datasets.datasets.vg_vqa_datasets import VGVQADataset
from lavis.datasets.datasets.gqa_datasets import GQADataset, GQAEvalDataset
from lavis.datasets.datasets.LN_datasets import LNDataset, LNEvalDataset


@registry.register_builder("coco_LN")
class COCOLNBuilder(BaseDatasetBuilder):
    train_dataset_cls = LNDataset
    eval_dataset_cls = LNEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_LN.yaml",
        # "eval": "configs/datasets/coco/eval_LN.yaml",
    }
