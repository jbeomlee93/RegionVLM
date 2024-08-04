import json
import jsonlines
import os
import spacy
import re
import re
import random


def get_points_from_time(traces, s_time, e_time):
    local_points = []
    for t1 in traces:
        for t2 in t1:
            if s_time <= t2['time_ms_since_epoch'] <= e_time:
                local_points.append(t2)
    
    return local_points
    

root_dir = '/path/to/LN'
uvo_jsonfile = [os.path.join(root_dir, 'UVO_sparse', 'UVO_sparse_train.jsonl'), os.path.join(root_dir, 'UVO_sparse', 'UVO_sparse_val.jsonl')]
ovis_jsonfile = [os.path.join(root_dir, 'OVIS', 'OVIS_train.jsonl')]
oops_jsonfile = [os.path.join(root_dir, 'oops', 'oops_val.jsonl'), os.path.join(root_dir, 'oops', 'oops_train.jsonl')]
kinetics_jsonfile = [os.path.join(root_dir, 'kinetics', 'kinetics_train.jsonl')]

save_dict = []

target_file = ovis_jsonfile + uvo_jsonfile + oops_jsonfile + kinetics_jsonfile

# target_file = kinetics_jsonfile

uvo_image_dir = os.path.join(root_dir, 'UVO_sparse', 'uvo_frames_sparese_recheck')
ovis_image_dir = os.path.join(root_dir, 'OVIS', 'train')
oops_image_dir_train = os.path.join(root_dir, 'oops', 'oops_dataset', 'oops_frames')
oops_image_dir_val = os.path.join(root_dir, 'oops', 'oops_dataset', 'oops_frames_val')
kinetics_image_dir = os.path.join(root_dir, 'kinetics', 'kinetics_frames')

for co_file in target_file:
    print(co_file)
    x = jsonlines.open(co_file)
    count = 0

    for idx, item in enumerate(x.iter()):
        count += 1
        # if count > 300:
        #     break
        if 'kinetics' in co_file:
            if not os.path.exists('/path/to/LN/kinetics/kinetics_frames/%s' % item['video_id']):
                continue
        for an_idx, an in enumerate(item['actor_narratives'][:-1]):
            traces = an['traces']
            time_alignment = an['time_alignment']
            caption = an['caption']
            total_time = traces[0][0]['time_ms_since_epoch']
            noun_segments = an['noun_segments']
            subject_word = caption[noun_segments[0][0]:noun_segments[0][1]]
            # if 'first' in caption.lower():
            #     continue
            # if 'another' in caption.lower():
            #     continue
            if subject_word == 'group' or subject_word == 'Group':
                continue
            target_ta = []  # -1: target ta
            if ' is ' in caption:
                for ta_idx, ta in enumerate(time_alignment):
                    target_ta.append(ta)
                    if ta['referenced_word'] == 'is':
                        target_ta.append(time_alignment[ta_idx])
                        break
                
            else:
                for ta_idx, ta in enumerate(time_alignment):
                    target_ta.append(ta)
                    if ta['referenced_word'] == subject_word.split(' ')[-1]:
                        target_ta.append(time_alignment[ta_idx])
                        break
            

            all_points = []
            for ta in target_ta:
        #         print()
                s_time = ta['start_ms'] + total_time
                e_time = ta['end_ms'] + total_time
                all_points = all_points + get_points_from_time(traces, s_time, e_time)

            img_point_pair = {}
            for p in all_points:
                if p['kf_idx'] not in img_point_pair.keys():
                    img_point_pair[p['kf_idx']] = []
                img_point_pair[p['kf_idx']].append([p['x'], p['y']])
#             print(img_point_pair)
            plot_images, plot_labels = [], []
            plot_images, plot_labels = [], []
            if len(img_point_pair.keys()) == 0:
                continue
            for kf_key in img_point_pair.keys():
                sent_to_points = {}

                # print(item['video_id'], item['keyframe_names'][kf_key])
                if 'oops' in co_file:
                    oops_img_name = item['video_id'] + '/' + item['keyframe_names'][kf_key] + '.png'
                    oops_img_name = oops_img_name.split('/0')[0] + '/' + str(int(oops_img_name.split('/')[-1].split('.png')[0])) + '.png'
                    if os.path.exists(os.path.join(oops_image_dir_train, oops_img_name)):
                        image_path = os.path.join(oops_image_dir_train, oops_img_name)
                    elif os.path.exists(os.path.join(oops_image_dir_val, oops_img_name)):
                        image_path = os.path.join(oops_image_dir_val, oops_img_name)
                    else:
                        image_path = None
                        print(item['video_id'], item['keyframe_names'][kf_key])
                        continue
                elif 'kinetics' in co_file:
                    image_name = item['video_id'] + '/' + item['keyframe_names'][kf_key] + '.png'
                    image_name = image_name.split('/0')[0] + '/' + str(int(image_name.split('/')[-1].split('.png')[0])) + '.png'
                    image_path = os.path.join(kinetics_image_dir, image_name)
                    if not os.path.exists(image_path):
                        print(image_path)
                        image_path = None
                        print(item['video_id'], item['keyframe_names'][kf_key])
                        
                        continue
                    else:
                        print("success!!")
                else:
                    if os.path.exists(os.path.join(uvo_image_dir, item['video_id'], item['keyframe_names'][kf_key]+'.png')):
                        image_path = os.path.join(uvo_image_dir, item['video_id'], item['keyframe_names'][kf_key]+'.png')
                    elif os.path.exists(os.path.join(ovis_image_dir, item['video_id'], item['keyframe_names'][kf_key]+'.jpg')):
                        image_path = os.path.join(ovis_image_dir, item['video_id'], item['keyframe_names'][kf_key]+'.jpg')
        
                    else:
                        image_path = None
                        print(item['video_id'], item['keyframe_names'][kf_key])
                        continue
                n_point = 20
                points = img_point_pair[kf_key]
                sel_points = [points[i] for i in [int((len(points)-1)/(n_point-1)*(j-1)) for j in range(1, n_point+1)]]
                sent_to_points[caption] = sel_points
            
                result_item = {'caption': caption, 'sent_to_points': sent_to_points, 'chunk_to_points': None, 'image_path': image_path}
                save_dict.append(result_item)
            # print(result_item)
with open("/path/to/lavis/coco/annotations/total_LN_chunk_sentence_ten_video_allvids_point20.json", "w") as json_file:

    json.dump(save_dict, json_file)




