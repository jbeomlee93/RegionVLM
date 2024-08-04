CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip2/eval/caption_coco_flant5xl_eval.yaml
