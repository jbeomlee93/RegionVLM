python -m torch.distributed.run --nproc_per_node=8 --master_port=29505 train.py --cfg-path lavis/projects/blip2/train/LN_ft.yaml
