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
        
        self.p_whole_sentence = 0.5
        self.cc3m_ann = None


    def select_point(self, points, k):
        sel_points = [points[i] for i in [int((len(points)-1)/(k-1)*(j-1)) for j in range(1, k+1)]]
        sel_points = [[int(item*100) for item in sublist] for sublist in sel_points]
        return sel_points


    def __getitem__(self, index):

        # TODO this assumes image input, not general enough

        if self.cc3m_ann is None:
            self.cc3m_ann = json.load(open('/path/to/lavis/coco/annotations/total_LN_cc3m.json'))
            self.len_cc3m = len(self.cc3m_ann)

        ann = self.annotation[index]
        image_path = os.path.join(ann["image_path"])
        
        k = 5

        if 'sent_to_patch' in ann.keys():
            image = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image)
            target_sent = random.sample(ann['sent_to_patch'].keys(), 1)[0]
            points = ann["sent_to_patch"][target_sent]
            sel_points = [points[i] for i in [int((len(points)-1)/(k-1)*(j-1)) for j in range(1, k+1)]]
            text_input = self.text_processor(str(sel_points)[1:-1].replace(',', ''))       
            text_output = self.text_processor(target_sent)
            # print(text_input, text_output)

        elif 'chunk_to_points' in ann.keys():
            if random.random() > self.p_whole_sentence:
                image = Image.open(image_path).convert("RGB")
                image = self.vis_processor(image)  
                target_sent = random.sample(ann['sent_to_points'].keys(), 1)[0]
                points = ann["sent_to_points"][target_sent]
                sel_points = self.select_point(points, k)
                text_input = self.text_processor(str(sel_points)[1:-1].replace(',', ''))       
                text_output = self.text_processor(target_sent)
            else:
                
                cc_idx = random.randint(0, self.len_cc3m-1)
                cc_ann = self.cc3m_ann[cc_idx]
                image = Image.open(cc_ann['image_path']).convert("RGB")
                image = self.vis_processor(image)
                target_sent = cc_ann['caption']
                text_input = self.text_processor('')       
                text_output = self.text_processor(target_sent)

                # target_sent = ann['caption']
                # if k == 5:
                #     sel_points = [[25, 25], [25, 75], [50, 50], [75, 25], [75, 75]]
                # else:
                #     print("erooororororoororororo")
                # text_input = self.text_processor(str(sel_points)[1:-1].replace(',', ''))       
                # text_output = self.text_processor(target_sent)


            
        else: # old version with only chunks
            
            target_word = random.sample(ann['sent_to_points'].keys(), 1)[0]
            points = ann["sent_to_points"][target_word]

            sel_points = self.select_point(points, k)


            # sel_points = [[float(format(item, '.2f')) for item in sublist] for sublist in sel_points]
            # print("=================")
            # print(sel_points)

            if random.random() > -1:
                text_input = self.text_processor('points: ' + str(sel_points)[1:-1].replace(',', '') + ', caption: ')       
                text_output = self.text_processor(target_word)
            else:
                text_input = self.text_processor('caption: ' + target_word + ', points: ')       
                text_output = self.text_processor(str(sel_points)[1:-1].replace(',', ''))

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
        # print("annnnnn", ann_paths)
        # print(vis_processor)
        self.img_ids = {}
        n = 0
        print("Its init!")
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough


        ann = self.annotation[index]
        image_path = os.path.join(ann["image_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        
        if 'chunk_to_points' in ann.keys():
            words = ann['sent_to_points'].keys()
            text_inputs = []
            text_outputs = []
            for w in words:
                points = ann["sent_to_points"][w]
                k = 5
                # print(k)
                sel_points = [points[i] for i in [int((len(points)-1)/(k-1)*(j-1)) for j in range(1, k+1)]]
                sel_points = [[int(item*100) for item in sublist] for sublist in sel_points]

                text_input = self.text_processor(str(sel_points)[1:-1].replace(',', ''))
                text_inputs.append(text_input)
            # text_input = self.text_processor('A photo of ')
        
                text_output = self.text_processor(w)
                text_outputs.append(text_output)
            sel_points = [[25, 25], [25, 75], [50, 50], [75, 25], [75, 75]]
            text_input = self.text_processor(str(sel_points)[1:-1].replace(',', ''))
            text_inputs.append(text_input)
            text_outputs.append('global')

        else:
            words = ann['sent_to_points'].keys()
            text_inputs = []
            text_outputs = []
            for w in words:
                points = ann["sent_to_points"][w]
                k = 5
                # print(k)
                sel_points = [points[i] for i in [int((len(points)-1)/(k-1)*(j-1)) for j in range(1, k+1)]]
                sel_points = [[int(item*100) for item in sublist] for sublist in sel_points]

                text_input = self.text_processor('Localized narratives: ' + str(sel_points)[1:-1].replace(',', '') + ', caption: ')
                text_inputs.append(text_input)
            # text_input = self.text_processor('A photo of ')
        
                text_output = self.text_processor(w)
                text_outputs.append(text_output)
            # print(text_input)
            # print(text_output)
            # caption = self.text_processor(ann["caption"])
            # print(caption)

        return {
            "image": image,
            "text_input": text_inputs,
            "text_output": text_outputs,
            "image_id": self.img_ids[ann["image_id"]],
        }


