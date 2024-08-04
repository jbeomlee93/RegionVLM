import argparse

import os


import cv2
import ipyplot
from PIL import Image

import numpy as np

import torch
from PIL import Image, ImageDraw
# setup device to use
from lavis.models import load_model_and_preprocess
import copy
from lavis.datasets.datasets.LN_datasets import LNEvalDataset, LNEvalDataset_RIS
import tqdm
from joblib import Parallel, delayed
# from mask_to_scribbles import mask_to_scribble
import random


def generate_random_points_from_mask(mask, num_points):
    # Find the coordinates of "on" pixels in the binary mask
    on_pixel_coords = np.argwhere(mask == 1)

    # Shuffle the coordinates to randomize the selection
    np.random.shuffle(on_pixel_coords)

    # Select the first num_points coordinates as random points
    random_points = on_pixel_coords[:num_points]
#     random_points = np.concatenate([np.exprandom_points[:, 1], random_points[:, 0]], axis=1)
    random_points = np.take(random_points, [1,0], axis=1)
    return random_points

def Compute_IoU(pred, target, cum_I, cum_U, mean_IoU=[]):

    if target.dtype != torch.bool:
        target = target.type(torch.bool).squeeze(0)

    I = torch.sum(torch.logical_and(pred, target))
    U = torch.sum(torch.logical_or(pred, target))

    if U == 0:
        this_iou = 0.0
    else:
        this_iou = I * 1.0 / U
    I, U = I, U


    cum_I += I
    cum_U += U
    # print(this_iou)
    mean_IoU.append(this_iou)

    return this_iou, mean_IoU, cum_I, cum_U

def select_point(points, k, isrand=False):
    if isrand:
        if len(points) < k:
            sel_points = [points[i] for i in [int((len(points)-1)/(k-1)*(j-1)) for j in range(1, k+1)]]
        else:
            sel_points = [points[i] for i in sorted(random.sample(range(len(points)), k))]
    else:
        sel_points = [points[i] for i in [int((len(points)-1)/(k-1)*(j-1)) for j in range(1, k+1)]]

    # sel_points = [points[i] for i in sorted(random.sample(range(len(points)), k))]
    # sel_points = [[int(item*100) for item in sublist] for sublist in sel_points]
    return sel_points

def process(item):
    
    idx_start, idx_end, gpu_id, args = item
    device = torch.device("cuda:%s" % (gpu_id)) if torch.cuda.is_available() else "cpu"

    model, vis_processors, text_processors = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
    dataset = LNEvalDataset_RIS(vis_processors['train'], text_processors['train'], split=split, dataset_name=dataset_name, splitBy=splitBy)

    model.load_checkpoint('/path/to/ckpoint_RegionVLM.pth')


    coco_img_dir = '/path/to/mscoco/images/train2014'
    mask_dir = '/path/to/SAM_proposals'
    n_rand_point = 10
    prefix_prompt = ''
    model.eval()
    cum_I, cum_U =0, 0
    m_IoU = []
    n_repeat = 5

    for idx in range(idx_start, idx_end):

        sample = dataset.__getitem__(idx)
        image = sample['image']
        image = image.unsqueeze(0).to(device)
        
        target = sample['annot']
        sentences = sample['sentence_raw']
        file_name = sample['file_name']
        height = sample['height']
        width = sample['width']
        mask_names = os.listdir(os.path.join(mask_dir, file_name.split('.jpg')[0]))

        for sent_idx in range(len(sentences)):
            sel_mask = None
            sel_loss = 1000000
            losses_pool = []
            masks_pool = []
            for m_name in mask_names:
                mask = Image.open(os.path.join(mask_dir, file_name.split('.jpg')[0], m_name))
                mask = np.array(mask)/255
                if mask.sum () / mask.shape[0] / mask.shape[1] < args.small_threshold:
                    continue
                threshold = args.threshold
                if mask[:2].sum() > threshold * mask.shape[0] or mask[-2:].sum() > threshold * mask.shape[0] or mask[:, :2].sum() > threshold * mask.shape[1] or mask[:, -2:].sum() > threshold * mask.shape[1]:
                    continue
                text_inputs = []
                for rep in range(n_repeat):

                    rand_points = generate_random_points_from_mask(mask, n_rand_point).astype(np.float32)

                    rand_points[:, 0] = rand_points[:, 0] / mask.shape[1]
                    rand_points[:, 1] = rand_points[:, 1] / mask.shape[0]
                    rand_points = (rand_points * 100).astype(np.int).tolist()
                    text_input = text_processors['train'](str(rand_points)[1:-1].replace(',', ''))

                    text_inputs.append(text_input)

                with torch.no_grad():
                    loss = model.forward({"image": torch.cat([image for i in range(n_repeat)], dim=0), "text_input": text_inputs, "text_output": [text_processors['train'](sentences[sent_idx]) for i in range(n_repeat)]}, reduction='none')
                    model.count = 0
                boundary = loss['loss'].shape[0] // n_repeat
                losses = loss['loss'].data.cpu()
                losses = [losses[boundary*i:boundary*(i+1)].mean().item() for i in range(n_repeat)]
                losses_pool.append(np.array(losses).min())
                masks_pool.append(mask)
                if len(masks_pool) == 20:
                    break
            if len(losses_pool) == 0:
                for m_name in mask_names:
                    mask = Image.open(os.path.join(mask_dir, file_name.split('.jpg')[0], m_name))
                    mask = np.array(mask)/255
                    if mask.sum () / mask.shape[0] / mask.shape[1] < 0.01:
                        continue
                    text_inputs = []
                    for rep in range(n_repeat):
                        rand_points = generate_random_points_from_mask(mask, n_rand_point).astype(np.float32)
                        rand_points[:, 0] = rand_points[:, 0] / mask.shape[1]
                        rand_points[:, 1] = rand_points[:, 1] / mask.shape[0]
                        rand_points = (rand_points * 100).astype(np.int).tolist()
                        text_input = text_processors['train'](str(rand_points)[1:-1].replace(',', ''))
                        text_inputs.append(text_input)
                    
                    with torch.no_grad():
                        loss = model.forward({"image": torch.cat([image for i in range(n_repeat)], dim=0), "text_input": text_inputs, "text_output": [text_processors['train'](sentences[sent_idx]) for i in range(n_repeat)]}, reduction='none')
                        model.count = 0
                    boundary = loss['loss'].shape[0] // n_repeat
                    losses = loss['loss'].data.cpu()
                    losses = [losses[boundary*i:boundary*(i+1)].mean().item() for i in range(n_repeat)]
                    losses_pool.append(np.array(losses).min())
                    masks_pool.append(mask)
                    if len(masks_pool) == 20:
                        break

            sel_idxs = sorted(range(len(losses_pool)), key=lambda i: losses_pool[i])[:1]
            sel_mask = masks_pool[sel_idxs[0]]
            sel_mask = np.clip(masks_pool[sel_idxs[0]], 0, 1)
            _, m_IoU, cum_I, cum_U = Compute_IoU(torch.Tensor(sel_mask), torch.Tensor(target), cum_I, cum_U, m_IoU)
            overall = cum_I * 100.0 / cum_U
            mean_IoU = torch.mean(torch.tensor(m_IoU)) * 100.0
            print("=================== %s / %s" % (idx, idx_end))
            print("RefCOCO val oIoU:", overall)
            print("RefCOCO val mIoU:", mean_IoU)
            print("===================")
    
    return cum_I, cum_U, m_IoU


parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    "--dataset_name",
    default="refcoco",
    type=str,
)
parser.add_argument(
    "--splitBy",
    default="unc",
    type=str,
)
parser.add_argument(
    "--split",
    default="val",
    type=str,
)
parser.add_argument(
    "--threshold",
    default=0.7,
    type=float,
)

parser.add_argument(
    "--small_threshold",
    default=0.04,
    type=float,
)

args = parser.parse_args()

print(args)
dataset_name = args.dataset_name
splitBy = args.splitBy
split = args.split

dataset = LNEvalDataset_RIS(None, None, split=split, dataset_name=dataset_name, splitBy=splitBy)
len_dataset = len(dataset)

# len_dataset = 20
sample_per_gpu = len_dataset // n_gpus
inputs = [[sample_per_gpu*i, sample_per_gpu*(i+1), i, args]for i in range(n_gpus)]
inputs[-1][1] = len_dataset
print(inputs)
output = Parallel(n_jobs=n_gpus)(delayed(process)(i) for i in inputs)
cum_I, cum_U = 0, 0
m_IoU = []
for o in output:
    cum_I += o[0]
    cum_U += o[1]
    m_IoU.append(o[2])
print(len(m_IoU))
m_IoU = sum(m_IoU, [])
print("=====================")
print("oIoU:", cum_I * 100.0 / cum_U)
print("mIoU:", torch.mean(torch.tensor(m_IoU)) * 100.0)
print(len(m_IoU))

write_file = open('RIS_evals.txt', 'a')
write_file.write('%s_%s_%s_%s_%f_%f_%f_%f\n' % (dataset_name, splitBy, split, 'ours', args.threshold
, args.small_threshold, cum_I * 100.0 / cum_U, torch.mean(torch.tensor(m_IoU)) * 100.0))
print(dataset_name, splitBy, split, 'ours')