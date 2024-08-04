import json
import jsonlines
import os
import spacy
import re

# import nltk
import re
# from pycorenlp import *
import random

def check_inside(tr, tc):
    if tc['start_time'] <= tr['t'] <= tc['end_time']:
        return True
    else:
        return False

def time_to_boxes(timed_caption, traces):
    return_dict = {} # utterance: boxes
    traces_flat = sum(traces, [])
    char_idx = 0
    # print(traces_flat)
    for tc_idx, tc in enumerate(timed_caption):
        for tr in traces_flat:
#             print(tc_idx, tc)
            if check_inside(tr, tc):
                if '%d-%d_%s' % (char_idx, char_idx + len(tc['utterance'])+1, tc['utterance']) not in return_dict.keys():
                    return_dict['%d-%d_%s' % (char_idx, char_idx + len(tc['utterance'])+1, tc['utterance'])] = []
                return_dict['%d-%d_%s' % (char_idx, char_idx + len(tc['utterance'])+1, tc['utterance'])].append([tr['x'], tr['y']])
        char_idx = char_idx + len(tc['utterance'])+1
    return return_dict


def sentence_to_narr(sentences, caption, word_to_points):
    # sentences: a set of chunks 
    sent_to_narr = {}
   
    done_caption = caption[:]

    for sent in sentences:
        if sent == '':
            continue
        print(sent)
        sent_to_narr[sent] = []
        start_idx = done_caption.find(sent)
        end_idx = start_idx + len(sent)
        done_caption = done_caption[:start_idx] + '-'*len(sent)+done_caption[end_idx:]
        for char_key in word_to_points.keys():
            start_now, end_now = char_key.split('_')[0].split('-')
            start_now, end_now = int(start_now), int(end_now)
            # print(start_now, end_now)
            if start_idx <= start_now <= end_idx:
                sent_to_narr[sent].append(word_to_points[char_key])
    for sent in list(sent_to_narr.keys()):
        if sent_to_narr[sent] == []:
            del(sent_to_narr[sent])
        else:
            sent_to_narr[sent] = sum(sent_to_narr[sent], [])

    return sent_to_narr
        # print(sent, caption[start_idx:end_idx])
        # print(start_idx)
        # if start_idx == -1:
        #     print(sent, caption)
    # narr = []
    # now_sentences = ''
    # print("==============================================")

    # for tc_idx, tc in enumerate(timed_caption):
    #     if 
    #     now_sentences += tc['utterance'] + ' '
    #     print(now_sentences.replace('.', '').replace(',', '').strip(), sent_to_narr.keys())
    #     if now_sentences.replace('.', '').replace(',', '').strip() in sent_to_narr.keys():
    #         print('*********', now_sentences)
    #         now_sentences = ''


def generate_random_points(box, k):
    """
    Generate K random points inside the given box.
    
    Args:
        box: A tuple (x_min, y_min, x_max, y_max) representing the bounding box.
        k: The number of random points to generate.
    
    Returns:
        A list of tuples representing the generated random points [(x1, y1), (x2, y2), ...].
    """
    x_min, y_min, x_max, y_max = box
    random_points = []
    
    for _ in range(k):
        x = round(random.uniform(x_min, x_max), 4)
        y = round(random.uniform(y_min, y_max), 4)
        random_points.append([x, y])
    
    return random_points


def mapping_index(idx, caption, timed_caption):
    pass

root_dir = '/path/to/lavis/LN'
coco_files = [os.path.join(root_dir, 'COCO', 'coco_train_localized_narratives-0000%d-of-00004.jsonl' % i) for i in range(4)]
openimages_files = [os.path.join(root_dir, 'OpenImages', 'open_images_train_v6_localized_narratives-0000%d-of-00010.jsonl' % i) for i in range(10)]
flickr_files = [os.path.join(root_dir, 'Flickr', 'flickr30k_train_localized_narratives.jsonl')]
ade_files = [os.path.join(root_dir, 'ADE', 'ade20k_train_localized_narratives.jsonl')]
nlp_spacy = spacy.load('en_core_web_lg')

mode = 'chunk'  # chunk | sentence
dataset = 'OpenImages' # OpenImages | COCO

save_dict = []
query_idx = 0

# target_file = coco_files if dataset == 'COCO' else openimages_files
target_file = ade_files + flickr_files + openimages_files + coco_files
# target_file = [os.path.join(root_dir, 'val', 'coco_val_localized_narratives.jsonl')]
# target_file = []

# target_file = [os.path.join(root_dir, 'COCO', 'coco_val_localized_narratives.jsonl')]
# target_file = [os.path.join(root_dir, 'COCO', 'coco_train_localized_narratives-0000%d-of-00004.jsonl' % i) for i in range(1)]

coco_image_train_dir = '/path/to/lavis/coco/images/train2014'
coco_image_val_dir = '/path/to/lavis/coco/images/val2014'
openimages_image_dir = '/path/to/lavis/LN/OpenImages'
flickr_image_dir = '/path/to/flickr30k_images'
ade_image_dir = '/path/to/lavis/LN/ADE/ADE20K_2021_17_01/images/training_images_full'
for co_file in target_file:
    print(co_file)
    x = jsonlines.open(co_file)

    for idx, img in enumerate(x.iter()):
        # print(idx)
        # if idx > 2000:
        #     break
        caption = img['caption']
        timed_caption = img['timed_caption']
        traces = img['traces']
        
        doc = nlp_spacy(caption)
        word_to_points = time_to_boxes(timed_caption, traces)
        
        try:
            if os.path.exists(os.path.join(coco_image_train_dir, 'COCO_train2014_000000%06d.jpg' % int(img['image_id']))):
                image_path = os.path.join(coco_image_train_dir, 'COCO_train2014_000000%06d.jpg' % int(img['image_id']))
            elif os.path.exists(os.path.join(coco_image_val_dir, 'COCO_val2014_000000%06d.jpg' % int(img['image_id']))):
                image_path = os.path.join(coco_image_val_dir, 'COCO_val2014_000000%06d.jpg' % int(img['image_id']))
        except:
            if os.path.exists(os.path.join(openimages_image_dir, 'images', 'train_%s' % img['image_id'][0], img['image_id']+'.jpg')):
                image_path = os.path.join(openimages_image_dir, 'images', 'train_%s' % img['image_id'][0], img['image_id']+'.jpg')
            elif os.path.exists(os.path.join(flickr_image_dir, img['image_id']+'.jpg')):
                image_path = os.path.join(flickr_image_dir, img['image_id']+'.jpg')
            elif os.path.exists(os.path.join(ade_image_dir, img['image_id']+'.jpg')):
                image_path = os.path.join(ade_image_dir, img['image_id']+'.jpg')
            else:
                print("no image error!")


        # sentences
        # print(caption)
        # remove_prefix = [' there is ', 'there are ', 'In this image ', 'In this picture ', 'I can see ', 'We can see ', 'in the background ', 'In the background ', 'we can see ', 'we can find ', 'we see ', 'In the image ', 'We see ', 'Threre are ', ' There is ']
        # # sentences = re.split('\. |\.|\,|\, |, and |there is |there are |In this image | In this picture |In this picture |I can see |We can see | in the background | In the background | we can see | we can find | we see |In the image', caption)
        # for rp in remove_prefix:
        #     caption = caption.replace(rp, '')
        
        sentences = re.split('\. |\.| and |\,|\, |, and', caption)
        sent_to_points = sentence_to_narr(sentences, caption, word_to_points)
        if len(sent_to_points.keys()) == 0:
            continue
        
        n_point = 20
        for kk in sent_to_points.keys():
            points = sent_to_points[kk]

            sel_points = [points[i] for i in [int((len(points)-1)/(n_point-1)*(j-1)) for j in range(1, n_point+1)]]
            sent_to_points[kk] = sel_points
        
        

        # chunk 
        # chunks = doc.noun_chunks
        # chunks = [str(s) for s in chunks if not ('picture' in str(s) or 'image' in str(s))]


        # chunk_to_points = sentence_to_narr(chunks, caption, word_to_points)
        # if len(chunk_to_points.keys()) == 0:
        #     continue
        
        
        # n_point = 10
        # for kk in chunk_to_points.keys():
        #     points = chunk_to_points[kk]

        #     sel_points = [points[i] for i in [int((len(points)-1)/(n_point-1)*(j-1)) for j in range(1, n_point+1)]]
        #     chunk_to_points[kk] = sel_points
        chunk_to_points=None
        item = {'sent_to_points': sent_to_points, 'image_path': image_path}
        # item = {'image_id': img['image_id'], 'query_idx': query_idx, 'caption': caption, 'sent_to_points': sent_to_points, 'chunk_to_points': chunk_to_points, 'image_path': image_path}
        print(item)
        save_dict.append(item)
        query_idx += 1

print("vg")
### visual genome
vg_dir = '/path/to/lavis/vg'
anno_file = os.path.join(vg_dir, 'annotations', 'train.json')
img_dir = os.path.join(vg_dir, 'image')
json_file = json.load(open(anno_file))
anns = json_file['annotations']
images = json_file['images']
images_dict = {}
img_to_ann = {}
for img in images:
    images_dict[img['id']] = img
    img_to_ann[img['id']] = []

for ann in anns:
    img_to_ann[ann['image_id']].append(ann)

n_point = 20

for img_id in images_dict.keys():
    anns = img_to_ann[img_id]
    image_meta = images_dict[img_id]
    if not os.path.exists(os.path.join(img_dir, str(img_id)+'.jpg')):
                print("nonono")
    image_path = os.path.join(img_dir, str(img_id)+'.jpg')
    sent_to_points = {}

    for ann in anns:
        box = ann['bbox']
        box = [box[0]/image_meta['width'], box[1]/image_meta['height'], box[2]/image_meta['width'], box[3]/image_meta['height']]
        if box[2]*box[3] > 0.5:
            continue
        box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        # box = [round(b, 4) for b in box]
        points = generate_random_points(box, n_point)

        caption = ann['caption']
        sel_points = [points[i] for i in [int((len(points)-1)/(n_point-1)*(j-1)) for j in range(1, n_point+1)]]
        sent_to_points[caption] = sel_points
    # print(sent_to_points)

    item = {'sent_to_points': sent_to_points, 'image_path': image_path}
    # print(item)
    save_dict.append(item)
    query_idx += 1

    # if query_idx > 100:
    #     break

        

with open("/path/to/lavis/coco/annotations/total_LN_chunk_sentence_point_20_image_vg.json", "w") as json_file:

    json.dump(save_dict, json_file)
