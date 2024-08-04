CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --master_port=29501 evaluate.py --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval.yaml
