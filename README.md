
# [Toward Interactive Regional Understanding in Vision-Large Language Models (NAACL'24)](https://arxiv.org/pdf/2403.18260)

by **Jungbeom Lee, Sanghyuk Chun, and Sangdoo Yun.** 


## Abstract
Recent Vision-Language Pre-training (VLP) models have demonstrated significant advancements. Nevertheless, these models heavily rely on image-text pairs that capture only coarse and global information of an image, leading to a limitation in their regional understanding ability. In this work, we introduce RegionVLM,
equipped with explicit regional modeling capabilities, allowing them to understand userindicated image regions. To achieve this, we design a simple yet innovative architecture, requiring no modifications to the model architecture or objective function. Additionally, we
leverage a dataset that contains a novel source of information, namely Localized Narratives, which has been overlooked in previous VLP research. Our experiments demonstrate that our
single generalist model not only achieves an interactive dialogue system but also exhibits superior performance on various zero-shot region understanding tasks, without compromising its ability for global image understanding.

## Installation

- We kindly refer to the offical implementation of [BLIP2](https://github.com/salesforce/LAVIS).

## Usage

#### Step 1. Prepare Dataset

- Download Referring Image Segmentation datasets at [here](https://github.com/lichengunc/refer)
- Download VCR dataset at [here](https://visualcommonsense.com/)
- Download VQA datasets at [here](https://github.com/salesforce/LAVIS)

#### Step 2. Prepare pre-trained model

- Pre-trained model used in this paper: [Download](https://drive.google.com/file/d/1iATjOtOMZo2P2Eecbn3naqy7YvFVcH_d/view?usp=sharing).

#### Step 3. Run Referring Image Segmentation (RIS) inference

- Run [SAM](https://github.com/facebookresearch/segment-anything) to obtain object proposals
- Run the command line for the inference:
```
python eval_RIS.py --dataset_name refcoco --split val --threshold 0.7 --small_threshold 0.04 --splitBy unc
```

#### Step 4. Run Visual Commonsense Reasoning (VCR) inference

- Run the command line for the inference:
```
python eval_VCR.py
```
#### Step 5. Run Visual Question Answering (VQA) inference

- Run the command lines to run the inference for each dataset
```
bash run_scripts/blip2/eval/eval_gqa_zeroshot_flant5xl.sh        # GQA
bash run_scripts/blip2/eval/eval_okvqa_zeroshot_flant5xl.sh      # OKVQA
bash run_scripts/blip2/eval/validate_vqa_zeroshot_flant5xl.sh    # VQAv2
```

## Acknowledgment
- This code is heavily borrowed from [BLIP2](https://github.com/salesforce/LAVIS).