import os
import sys
# sys.path.append('../')
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from PIL import Image, ImageDraw
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
from lavis.models import load_model_and_preprocess
import copy
from r2c.dataloaders.my_vcr import VCR, VCRLoader
import cv2
import numpy as np
import re
from joblib import Parallel, delayed


def generate_random_points_from_mask(mask, num_points, text_processors):
    # Find the coordinates of "on" pixels in the binary mask
    on_pixel_coords = np.argwhere(mask == 1)
    
    # Shuffle the coordinates to randomize the selection
    np.random.shuffle(on_pixel_coords)
    
    # Select the first num_points coordinates as random points
    rand_points = on_pixel_coords[:num_points].astype(np.float32)
    rand_points = np.take(rand_points, [1,0], axis=1)
    
    rand_points[:, 0] = rand_points[:, 0] / mask.shape[1]
    rand_points[:, 1] = rand_points[:, 1] / mask.shape[0]
    rand_points = (rand_points * 100).astype(np.int).tolist()
    text_input = text_processors['train'](str(rand_points)[1:-1].replace(',', ''))
#     text_input = ''
    return text_input

def process(item):
    idx_start, idx_end, gpu_id = item
    device = torch.device("cuda:%s" % (gpu_id)) if torch.cuda.is_available() else "cpu"

    model, vis_processors, text_processors = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
    vcr = VCR(split='val', mode='answer', vis_processor=vis_processors['train'], text_processor=text_processors['train'])

    model.load_checkpoint('/path/to/ckpoint_RegionVLM.pth')

    total = 0
    correct_1, correct_2, correct_3 = 0,0,0

    for idx in range(idx_start,idx_end):
        total += 1
        item = vcr.__getitem__(idx)
        masks = []
        for m in item['masks']:
            masks.append(Image.fromarray((m*255).astype('uint8')))

        #### QA
        ans_string = ''
        for ans_idx in range(1, 5):
            ans_string += ' %s. ' % ans_idx
            ans_string += ' '.join([str(i) for i in item['answer'][ans_idx-1]])
        question_string = ' '.join([str(i) for i in item['question']]).strip() + ans_string
        question_string = 'Question: '+question_string + ' Answer: '

        obj_indices_start = [m.start() for m in re.finditer('\[', question_string)]
        obj_indices_end = [m.start()+1 for m in re.finditer('\]', question_string)]
        
        obj_indices = []
        start_obj = None
        for o in range(len(obj_indices_start)):
            cands = question_string[obj_indices_start[o]:obj_indices_end[o]]
            for c in cands:
                if c.isnumeric():
                    obj_indices.append(int(c))
                    if start_obj is None:
                        start_obj = int(c)
        
        rand_points = {}
        for o_idx in list(set(obj_indices)):
            points = generate_random_points_from_mask(item['masks'][o_idx], 10, text_processors=text_processors)        
            rand_points[o_idx] = points
        try:
            text_input = rand_points[start_obj]
        except:
            text_input = ''
        
        input_samples = {"image": item['image'].unsqueeze(0).to(device), 
                        "text_input": text_input, 
                        "question": question_string,
                        "obj_indices": obj_indices,
                        "rand_points": rand_points,
                        }
        x = model.predict_answers_describe_first(input_samples, short=False)
        try:
            pred_answer = re.findall(r'\d+', x[0])[0]
        except:
            pred_answer = '1'
        if str(int(vcr.item_now['answer_label'])+1) == pred_answer:
            correct_1 += 1


        #### QA->R
        
        ans_string = ''
        question_string = 'Question: '+' '.join([str(i) for i in item['question']]).strip() + ' Answer: ' + ' '.join([str(i) for i in item['answer'][vcr.item_now['answer_label']]]) + ' Why? '
        
        for ans_idx in range(1, 5):
            ans_string += ' %s. ' % ans_idx
            ans_string += ' '.join([str(i) for i in vcr.item_now['rationale_choices'][ans_idx-1]])
        
        question_string = question_string + ans_string
        
        obj_indices_start = [m.start() for m in re.finditer('\[', question_string)]
        obj_indices_end = [m.start()+1 for m in re.finditer('\]', question_string)]
        
        obj_indices = []
        start_obj = None
        for o in range(len(obj_indices_start)):
            cands = question_string[obj_indices_start[o]:obj_indices_end[o]]
            for c in cands:
                if c.isnumeric():
                    obj_indices.append(int(c))
                    if start_obj is None:
                        start_obj = int(c)
                
        
        rand_points = {}
        for o_idx in list(set(obj_indices)):
            points = generate_random_points_from_mask(item['masks'][o_idx], 10, text_processors=text_processors)
            
            rand_points[o_idx] = points
        try:
            text_input = rand_points[start_obj]
        except:
            text_input = ''

        input_samples = {"image": item['image'].unsqueeze(0).to(device), 
                        "text_input": text_input, 
                        "question": question_string,
                        "obj_indices": obj_indices,
                        "rand_points": rand_points,
                        }
        x = model.predict_answers_describe_first(input_samples, short=False)
        try:
            pred_rationale = re.findall(r'\d+', x[0])[0]
        except:
            pred_rationale = '1'
        if str(int(vcr.item_now['rationale_label'])+1) == pred_rationale:
            correct_2 += 1
        

        #### Q->AR
        ans_string = ''
        try:
            question_string = 'Question: '+' '.join([str(i) for i in item['question']]).strip() + ' Answer: ' + ' '.join([str(i) for i in item['answer'][int(pred_answer)-1]]) + ' Why? '
        except:
            question_string = 'Question: '+' '.join([str(i) for i in item['question']]).strip() + ' Answer: ' + ' '.join([str(i) for i in item['answer'][int(1)-1]]) + ' Why? ' # anyway wrong sample
 
        for ans_idx in range(1, 5):
            ans_string += ' %s. ' % ans_idx
            ans_string += ' '.join([str(i) for i in vcr.item_now['rationale_choices'][ans_idx-1]])
        
        question_string = question_string + ans_string

        obj_indices_start = [m.start() for m in re.finditer('\[', question_string)]
        obj_indices_end = [m.start()+1 for m in re.finditer('\]', question_string)]
        
        obj_indices = []
        start_obj = None
        for o in range(len(obj_indices_start)):
            cands = question_string[obj_indices_start[o]:obj_indices_end[o]]
            for c in cands:
                if c.isnumeric():
                    obj_indices.append(int(c))
                    if start_obj is None:
                        start_obj = int(c)
                
        
        rand_points = {}
        for o_idx in list(set(obj_indices)):
            points = generate_random_points_from_mask(item['masks'][o_idx], 10, text_processors=text_processors)
            
            rand_points[o_idx] = points
        try:
            text_input = rand_points[start_obj]
        except:
            text_input = ''
        input_samples = {"image": item['image'].unsqueeze(0).to(device), 
                        "text_input": text_input, 
                        "question": question_string,
                        "obj_indices": obj_indices,
                        "rand_points": rand_points,
                        }
        x = model.predict_answers_describe_first(input_samples, short=False)
        try:
            pred_rationale = re.findall(r'\d+', x[0])[0]
        except:
            pred_rationale = '1'
        if str(int(vcr.item_now['rationale_label'])+1) == pred_rationale and str(int(vcr.item_now['answer_label'])+1) == pred_answer:
            correct_3 += 1
        if total % 100 == 0:
            print(idx, correct_1/total, correct_2/total, correct_3/total)

    return correct_1, correct_2, correct_3, total



dataset = VCR(split='val', mode='answer', vis_processor=None, text_processor=None)

len_dataset = len(dataset)

n_gpus = 8
sample_per_gpu = len_dataset // n_gpus
inputs = [[sample_per_gpu*i, sample_per_gpu*(i+1), i]for i in range(n_gpus)]
inputs[-1][1] = len_dataset

print(inputs)
output = Parallel(n_jobs=n_gpus)(delayed(process)(i) for i in inputs)
correct_1, correct_2, correct_3 = 0.0, 0.0, 0.0
total = 0
for o in output:
    correct_1 += o[0]
    correct_2 += o[1]
    correct_3 += o[2]
    total += o[3]
print(total)
Q_A = correct_1 / total
QA_R = correct_2 / total
Q_AR = correct_3 / total
print(Q_A, QA_R, Q_AR)
write_file = open('VCR_evals.txt', 'a')
write_file.write('%f_%f_%f\n' % (Q_A, QA_R, Q_AR))
