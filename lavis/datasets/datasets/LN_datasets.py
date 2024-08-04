"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import random
import json
import torch
import tarfile
import traceback
import re   
import json
import os

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
from torchdata.datapipes.iter import TarArchiveLoader
from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes import functional_datapipe

from loguru import logger
from typing import Any, Dict, Iterator, List, Union
from torchdata.datapipes.iter import IterableWrapper, FSSpecFileOpener
# from refer import REFER
from lavis.datasets.datasets.refer import REFER
from torch.utils.data import Dataset
import numpy as np

import cv2
from PIL import ImageFilter, ImageDraw

def pathsplit(p):
    p = p.replace("\\", "/")
    if "." not in p:
        return p, ""
    match = re.search(r"^(.*?)(\.[^/]*)$", p)
    if not match:
        return p, ""
    prefix, suffix = match.groups()
    return prefix, suffix



def get_bounding_box(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(max_contour)
        return (x, y, x + w, y + h)  
    else:
        return None

# bounding_box = get_bounding_box(binary_mask)

def vis_prompt(img, mask, method):
    box = get_bounding_box(mask)
    
    if method == 'caption_anything':
        draw_img = np.array(img)
        draw_img = Image.fromarray(draw_img[box[1]:box[3], box[0]:box[2]])

    elif method == 'FGVP':
        draw_img = img.filter(ImageFilter.GaussianBlur(10))
        draw_img = Image.fromarray(np.array(draw_img) * (1-mask[:, :, None]) + np.array(img) * mask[:, :, None])

    elif method == 'red_circle':
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        draw.ellipse(box, fill=None, outline='red', width=10)
    
    return draw_img


@functional_datapipe("emptysuffix_webdataset")
class WebDataset(IterDataPipe[Dict]):
    def __init__(self, source_datapipe: IterDataPipe[List[Union[Dict, List]]]) -> None:
        self.source_datapipe: IterDataPipe[List[Union[Dict, List]]] = source_datapipe

    def __iter__(self) -> Iterator[Dict]:
        sample: Dict[str, Any] = {}
        current = ""
        for path, data in self.source_datapipe:
            assert isinstance(path, str), path
            prefix, suffix = pathsplit(path)
            if suffix == "":
                # files with empty suffixes can be used for metadata
                # they cannot be used for data since they wouldn't have a key
                # coyo는 suffix가 없는채로 이미지를 저장해서 임의로 nosuffix로 달아줌
                suffix = ".nosuffix"
            if prefix != current:
                if current != "":
                    yield sample
                sample = {}
                current = prefix
                sample["__key__"] = current
            sample[suffix] = data
        if sample != {}:
            yield sample

    def __len__(self) -> int:
        return len(self.source_datapipe)


@functional_datapipe("continued_load_from_tar")
class ContinuedTarArchiveLoaderIterDataPipe(TarArchiveLoader):
    def __iter__(self):
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                # typing.cast is used here to silence mypy's type checker
                tar = tarfile.open(fileobj=data_stream, mode=self.mode)
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        logger.warning(f"failed to extract file {tarinfo.name} from source tarfile {pathname}")
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
                    yield inner_pathname, StreamWrapper(extracted_fobj)  # type: ignore[misc]
            except Exception as e:
                logger.warning(f"{pathname=} {traceback.format_exc()=}")
                continue


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class LNDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        print("annnnnn", ann_paths)
        # print(vis_processor)
        # self.img_ids = {}
        # n = 0
        # for ann in self.annotation:
        #     img_id = ann["image_id"]
        #     if img_id not in self.img_ids.keys():
        #         self.img_ids[img_id] = n
        #         n += 1
        
        self.p_whole_sentence = 0.4
        self.cc3m_ann = None
        self.laion_115m_pipe_data = self.init_webdataset()
        self.laion_115m_pipe = iter(self.laion_115m_pipe_data)
        self.vg_idxs = []
        for idx, item in enumerate(self.annotation):
            if 'vg' in item['image_path'] or 'OVIS' in item['image_path'] or 'oops' in item['image_path'] or 'UVO' in item['image_path'] or 'kinetics' in item['image_path']:
            # if 'OVIS' in item['image_path'] or 'oops' in item['image_path'] or 'UVO' in item['image_path'] or 'kinetics' in item['image_path']:
                self.vg_idxs.append(idx)
        # self.vg_start = self.vg_idxs[0]
        self.delta_max = 0.1
        self.interleave = False
        self.coco_global = True
        if self.coco_global:
            self.coco_annot = json.load(open('/path/to/lavis/coco/annotations/coco_karpathy_train.json'))
        self.p_indicate_caption = -1

    def init_webdataset(self):
        buffer_size = 10000
        batch_size = 1
        rank = torch.distributed.get_rank()
        tar_len = 10488//8
        # print("rank", torch.distributed.get_rank())
        laion_115m_urls = [f"s3://vl-data/laion115m_blip/{i:05d}.tar" for i in range(tar_len*rank, tar_len*(rank+1))]

        laion_115m_pipe = IterableWrapper(laion_115m_urls)
        laion_115m_pipe = laion_115m_pipe.shuffle(buffer_size=buffer_size)
        laion_115m_pipe = FSSpecFileOpener(laion_115m_pipe, mode="rb")
        laion_115m_pipe = ContinuedTarArchiveLoaderIterDataPipe(laion_115m_pipe)
        laion_115m_pipe = WebDataset(laion_115m_pipe)
        laion_115m_pipe = laion_115m_pipe.batch(batch_size)

        return laion_115m_pipe

    def select_point(self, points, k):
        if len(points) < k:
            points = [points[i] for i in [int((len(points)-1)/(k-1)*(j-1)) for j in range(1, k+1)]]
        sel_points = [points[i] for i in sorted(random.sample(range(len(points)), k))]
        sel_points = [[int(item*100) for item in sublist] for sublist in sel_points]
        return sel_points

    def data_augmentation(self, img, points):
        img_array = np.array(img)
        h, w, _ = img_array.shape
        points = np.array(points)
        points = np.clip(points, 0, 1)
        tol_w = [min(self.delta_max, points[:, 0].min()), min(self.delta_max, 1-points[:, 0].max())]
        tol_h = [min(self.delta_max, points[:, 1].min()), min(self.delta_max, 1-points[:, 1].max())]
        wd1, wd2 = random.uniform(0, tol_w[0]), random.uniform(0, tol_w[1])
        hd1, hd2 = random.uniform(0, tol_h[0]), random.uniform(0, tol_h[1])

        points[:, 0] = (points[:, 0]-wd1) / (1-wd1-wd2)
        points[:, 1] = (points[:, 1]-hd1) / (1-hd1-hd2)

        wd1, wd2 = int(wd1 * w), int(wd2 * w)
        hd1, hd2 = int(hd1 * h), int(hd2 * h) 
        img = Image.fromarray(img_array[hd1:(h-hd2), wd1:(w-wd2), :])
        points = points.tolist()

        return img, points



    def __getitem__(self, index):

        ann = self.annotation[index]
        image_path = os.path.join(ann["image_path"])
        
        k = 10

        rand_score = random.random()

        if rand_score > self.p_whole_sentence:

            image = Image.open(image_path).convert("RGB")

            target_sent = random.sample(ann['sent_to_points'].keys(), 1)[0]
            points = ann["sent_to_points"][target_sent]
            
            image, points = self.data_augmentation(image, points)
            image = self.vis_processor(image)  
            sel_points = self.select_point(points, k)
            # sel_points = [int(item*100) for item in points]
            text_input = self.text_processor(str(sel_points)[1:-1].replace(',', ''))       
            text_output = self.text_processor(target_sent)
        else:
            # global
            rand_score = random.random()
            if not self.coco_global:
                rand_score = 1

            if rand_score < 0.1:
                # coco
                item = random.sample(self.coco_annot, 1)[0]
                image = Image.open(os.path.join('/path/to/lavis/coco/images', item['image'])).convert("RGB")
                image = self.vis_processor(image) 
                text_input = self.text_processor('')
                text_output = self.text_processor(item['caption'])
            else:
                try:
                    item = next(self.laion_115m_pipe)[0]
                except:
                    print("change????????")
                    self.laion_115m_pipe = iter(self.laion_115m_pipe_data)
                    item = next(self.laion_115m_pipe)[0]

                image = Image.open(item[".jpg"]).convert("RGB")
                image = self.vis_processor(image)
                kvs_json = json.load(item[".json"])
                target_sent= kvs_json["caption"]
                text_input = self.text_processor('')     
                text_output = self.text_processor(target_sent)
                # prefix = ''


        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
            # "image_id": self.img_ids[ann["image_id"]],
        }


class LNEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        print("Its init!")


    def __getitem__(self, index):

        # TODO this assumes image input, not general enough


        ann = self.annotation[index]
        image_path = os.path.join(ann["image_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        
        # if 'chunk_to_points' in ann.keys():
        words = ann['sent_to_points'].keys()
        text_inputs = []
        text_outputs = []
        for w in words:
            points = ann["sent_to_points"][w]
            k = 10
            sel_points = [points[i] for i in [int((len(points)-1)/(k-1)*(j-1)) for j in range(1, k+1)]]
            sel_points = [[int(item*100) for item in sublist] for sublist in sel_points]

            text_input = self.text_processor(str(sel_points)[1:-1].replace(',', ''))
            text_inputs.append(text_input)

            text_output = self.text_processor(w)
            text_outputs.append(text_output)

        return {
            "image": image,
            "text_input": text_inputs,
            "text_output": text_outputs,
            "ann": ann
            # "image_id": self.img_ids[ann["image_id"]],
        }


class LNEvalDataset_RIS(Dataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, dataset_name, split='val', splitBy='unc', refer_data_root='/path/to/refseg/dataset/'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        print("Its init!")
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.dataset_name = dataset_name
        self.split = split
        self.splitBy = splitBy
        self.refer_data_root = refer_data_root
        self.refer = REFER(refer_data_root, dataset_name, splitBy)
        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.sentence_raws = []
        for r in ref_ids:
            ref = self.refer.Refs[r]
            sentence_raw_for_ref = []
            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                sentence_raw_for_ref.append(sentence_raw)
            self.sentence_raws.append(sentence_raw_for_ref)



    def __len__(self):
        return len(self.ref_ids)


    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]
        # print(os.path.join(self.refer.IMAGE_DIR, this_img['file_name']))
        # print(os.path.exists(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])))
        image = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
        # print(np.array(image))

        image = self.vis_processor(image)
        ref = self.refer.loadRefs(this_ref_id)
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])

        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")


        sentence_raw = self.sentence_raws[index]


        return {'image': image, 'annot': np.asarray(annot), 'sentence_raw': sentence_raw, 'file_name': this_img['file_name'], 'height': np.asarray(image).shape[0], 'width': np.asarray(image).shape[1]}


if __name__ == "__mai∂n__":
    dataset = LNEvalDataset_RIS(None, None, dataset_name='refcoco', splitBy='unc')
    dataset.__getitem__(0)
    dataset.__getitem__(1)
    dataset.__getitem__(2)
