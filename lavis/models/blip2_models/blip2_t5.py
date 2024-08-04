"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
import random

@registry.register_model("blip2_t5")
class Blip2T5(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xl_vitL: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xl_vitL": "configs/models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
        "LN_coco_flant5xl": "configs/models/blip2/blip2_LN_flant5xl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()



        self.qformer_text_input = True
        self.tokenizer = self.init_tokenizer()
        self.freeze_qformer = False

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        print("freeze???", freeze_vit)
        print("qformer freeze???", self.freeze_qformer)
        print("qformer text input??", self.qformer_text_input)
        if freeze_vit:
            # print("freeze")
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        
        if self.freeze_qformer:
            self.Qformer, self.query_tokens, self.add_query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features, n_add_tokens=384
            )
        else:
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
            self.add_query_tokens = None

        if not self.qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        


        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        if self.freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            for param in self.t5_proj.parameters():
                param.requires_grad = False
            for param in self.ln_vision.parameters():
                param.requires_grad = False
            logging.info("freeze vision encoder")

        self.max_txt_len = max_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.count = 0
        print("self.add_query_tokens", self.add_query_tokens)
        self.ignore_null = True
        self.indicate_caption = False

    def forward(self, samples, reduction='mean'):
        

        if True:


            image = samples["image"]
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # 8x32x768
            # if self.add_query_tokens is not None:
            #     add_query_tokens = self.add_query_tokens.expand(image_embeds.shape[0], -1, -1)

            if self.qformer_text_input:
                # print(samples["text_input"])
                # print(samples["text_input"])
                text_Qformer = self.tokenizer(
                    samples["text_input"],
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)

                if True:
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

                    query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    # print(query_output.last_hidden_state.shape)
                    query_output.last_hidden_state = query_output.last_hidden_state[:, :32]
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            

            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

            if True:
                prefix = ['' for i in range(len(samples["text_input"]))]
            with self.maybe_autocast(dtype=torch.float32):
            # if True:
                input_tokens = self.t5_tokenizer(
                    # samples["text_input"],
                    prefix,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                output_tokens = self.t5_tokenizer(
                    samples["text_output"],
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)


                # print(encoder_atts)
                targets = output_tokens.input_ids.masked_fill(
                    output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
                )       
                inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)

                inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)



                outputs = self.t5_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    decoder_attention_mask=output_tokens.attention_mask,
                    return_dict=True,
                    labels=targets,
                    reduction=reduction,
                )

                loss = outputs.loss

                self.count += 1

                if self.count % 50 == 0:
                # if True:
                    outputs = self.t5_model.generate(
                        inputs_embeds=inputs_embeds[:2],
                        attention_mask=encoder_atts[:2],
                        do_sample=False,
                        top_p=0.9,
                        temperature=1,
                        num_beams=5,
                        max_new_tokens=30,
                        min_length=1,
                        repetition_penalty=1.0,
                        length_penalty=1.0,
                        num_return_sequences=1,
                    )
                    output_text = self.t5_tokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )

                    print(self.count, output_text)
                    print(samples["text_output"][:2])
                    if torch.distributed.get_rank() == 0:
                        print(image.shape)


                return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        qformer_text_input_eval=True,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        # print(samples)
        if "text_input" not in samples.keys():
            samples["text_input"] = ['' for i in range(image.shape[0])]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input and qformer_text_input_eval:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output.last_hidden_state = query_output.last_hidden_state[:, :32]
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        # if "prompt" in samples.keys():
        #     prompt = samples["prompt"]
        # else:
        #     prompt = self.prompt

        # if isinstance(prompt, str):
        #     prompt = [prompt] * image.size(0)
        # else:
        #     assert len(prompt) == image.size(
        #         0
        #     ), "The number of prompts must be equal to the batch size."

        # input_tokens = self.t5_tokenizer(
        #     prompt, padding="longest", return_tensors="pt"
        # ).to(image.device)

        # encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer(
                # samples["text_input"],
                ['' for i in range(len(samples["text_input"]))],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)



            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            print(output_text)

        return output_text

    def generate_random_points(self, box, k):
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

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        dense=False,
        **kwargs
    ):
        image = samples["image"]
        print(image.shape)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)



        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]
        
        
        # print(text_input)

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.float32):
            
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)


            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)
        # print(text_input, output_text)
        return output_text

    def predict_answers_interactive(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        dense=False,
        short=False,
        **kwargs
    ):

        image = samples["image"]
        question = samples["question"]
        obj_indices = samples["obj_indices"]
        rand_points = samples["rand_points"]

        # print(image.shape)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        inputs_t5_objs = {}
        for o_idx in rand_points.keys():
            text_Qformer = self.tokenizer(
                rand_points[o_idx],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output.last_hidden_state = query_output.last_hidden_state[:, :32]
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            inputs_t5_objs[o_idx] = inputs_t5
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        

        input_tokens = self.t5_tokenizer(
            samples["question"], padding="longest", return_tensors="pt"
        ).to(image.device)

        # encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.float32):
            # find 784 ("[") and 908 ("]")
            # print(input_tokens.input_ids)
            start_idxs = (input_tokens.input_ids[0]==784).nonzero(as_tuple=True)[0].tolist()
            end_idxs = (input_tokens.input_ids[0]==908).nonzero(as_tuple=True)[0].tolist()
            start_idxs = [int(s) for s in start_idxs]
            end_idxs = [int(s)+1 for s in end_idxs]
            # start_idxs = [0] + start_idxs
            print(start_idxs)
            print(end_idxs)
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)

            print(inputs_embeds.shape)
            start_idxs = start_idxs + [inputs_embeds.shape[1]+1]

            # for key in inputs_t5_objs.keys():
            #     print(inputs_t5_objs[key].shape)
            input_embeds_with_qformer = [inputs_embeds[:, :start_idxs[0]]]
            for s_idx in range(len(start_idxs)-1):
                obj_idx_now = obj_indices[s_idx]
                obj_query_now = inputs_t5_objs[obj_idx_now]

                input_embeds_with_qformer.append(obj_query_now)
                # input_embeds_with_qformer.append(inputs_embeds[:, (start_idxs[s_idx]+3):start_idxs[s_idx+1]])
                input_embeds_with_qformer.append(inputs_embeds[:, (end_idxs[s_idx]):start_idxs[s_idx+1]])
            # print("===================")
            for xxx in input_embeds_with_qformer:
                print(xxx.shape)
            inputs_embeds = torch.cat(input_embeds_with_qformer, dim=1)
            # print(input_embeds.shape)

            # inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            # print("only text is used !!")
            # print(inputs_embeds.shape)
            # inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            if short:
                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,     # if batch is 1, all 1 
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=max_len,
                    min_length=min_len,
                    length_penalty=length_penalty,
                )
            else:
                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,   # if batch is 1, all 1 
                    do_sample=False,
                    top_p=0.9,
                    temperature=1,
                    num_beams=num_beams,
                    max_new_tokens=30,
                    min_length=1,
                    repetition_penalty=1,
                    length_penalty=1,
                    num_return_sequences=1,
                )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        # if self._apply_lemmatizer:
        #     output_text = self._lemmatize(output_text)
        # print(text_input, output_text)
        return output_text


    def predict_answers_qformer_first(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        dense=False,
        short=False,
        **kwargs
    ):

        image = samples["image"]
        question = samples["question"]
        obj_indices = samples["obj_indices"]
        rand_points = samples["rand_points"]

        # print(image.shape)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        inputs_t5_objs = {}

        text_Qformer = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

        query_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=None,
            return_dict=True,
        )
        query_output.last_hidden_state = query_output.last_hidden_state[:, :32]
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        # for o_idx in rand_points.keys():
        #     text_Qformer = self.tokenizer(
        #         rand_points[o_idx],
        #         padding='longest',
        #         truncation=True,
        #         max_length=self.max_txt_len,
        #         return_tensors="pt",
        #     ).to(image.device)
        #     query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        #     Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

        #     query_output = self.Qformer.bert(
        #         text_Qformer.input_ids,
        #         attention_mask=Qformer_atts,
        #         query_embeds=query_tokens,
        #         encoder_hidden_states=image_embeds,
        #         encoder_attention_mask=image_atts,
        #         return_dict=True,
        #     )
        #     query_output.last_hidden_state = query_output.last_hidden_state[:, :32]
        #     inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        #     inputs_t5_objs[o_idx] = inputs_t5
        #     atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        # print("prefix attached!!!")
        # prefix_tokens = self.t5_tokenizer(
        #     prompt, padding="longest", return_tensors="pt"
        # ).to(image.device)

        input_tokens = self.t5_tokenizer(
            samples["question"], padding="longest", return_tensors="pt"
        ).to(image.device)

        # encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)


        with self.maybe_autocast(dtype=torch.float32):
            # prefix_embeds = self.t5_model.encoder.embed_tokens(prefix_tokens.input_ids)
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            # inputs_embeds = torch.cat([prefix_embeds, inputs_t5, inputs_embeds], dim=1)

            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)
        # print(text_input, output_text)
        return output_text



    def predict_answers_describe_first(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        dense=False,
        short=False,
        **kwargs
    ):

        image = samples["image"]
        question = samples["question"]
        obj_indices = samples["obj_indices"]
        rand_points = samples["rand_points"]

        # print(image.shape)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        inputs_t5_objs = {}
        for o_idx in rand_points.keys():
            text_Qformer = self.tokenizer(
                rand_points[o_idx],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output.last_hidden_state = query_output.last_hidden_state[:, :32]
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            
            inputs_t5_objs[o_idx] = inputs_t5
        
        # for p in prefix_string:
        #     print(p.shape)
        

        input_tokens = self.t5_tokenizer(
            samples["question"], padding="longest", return_tensors="pt"
        ).to(image.device)

        # prefix_string = []

        with self.maybe_autocast(dtype=torch.float32):
            start_string = self.t5_tokenizer(
                "Let", padding="longest", return_tensors="pt"
            ).to(image.device)
            # print(start_string.input_ids)
            start_string.input_ids = start_string.input_ids[:, :-1]
            # print(start_string.input_ids)

            start_embed = self.t5_model.encoder.embed_tokens(start_string.input_ids)
            prefix_string = [start_embed]
            for o_idx in rand_points.keys():
                tokenize_string = self.t5_tokenizer(
                    "[%s] be" % o_idx, padding="longest", return_tensors="pt"
                ).to(image.device)
                # print(tokenize_string.input_ids)
                tokenize_string.input_ids = tokenize_string.input_ids[:, :-1]
                # print(tokenize_string.input_ids)

                string_embed = self.t5_model.encoder.embed_tokens(tokenize_string.input_ids)

                prefix_string.append(string_embed)
                prefix_string.append(inputs_t5_objs[o_idx])
            # for p in prefix_string:
            #     print(p.shape)
            
            q_string = self.t5_tokenizer(
                question, padding="longest", return_tensors="pt"
            ).to(image.device)
            q_embed = self.t5_model.encoder.embed_tokens(q_string.input_ids)
            # print(q_embed.shape)
            inputs_embeds = torch.cat(prefix_string + [q_embed], dim=1)
            # print(inputs_embeds.shape)

            # # find 784 ("[") and 908 ("]")
            # # print(input_tokens.input_ids)
            # start_idxs = (input_tokens.input_ids[0]==784).nonzero(as_tuple=True)[0].tolist()
            # end_idxs = (input_tokens.input_ids[0]==908).nonzero(as_tuple=True)[0].tolist()
            # start_idxs = [int(s) for s in start_idxs]
            # end_idxs = [int(s)+1 for s in end_idxs]
            # # start_idxs = [0] + start_idxs
            # print(start_idxs)
            # print(end_idxs)
            # inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)

            # print(inputs_embeds.shape)
            # start_idxs = start_idxs + [inputs_embeds.shape[1]+1]

            # # for key in inputs_t5_objs.keys():
            # #     print(inputs_t5_objs[key].shape)
            # input_embeds_with_qformer = [inputs_embeds[:, :start_idxs[0]]]
            # for s_idx in range(len(start_idxs)-1):
            #     obj_idx_now = obj_indices[s_idx]
            #     obj_query_now = inputs_t5_objs[obj_idx_now]

            #     input_embeds_with_qformer.append(obj_query_now)
            #     # input_embeds_with_qformer.append(inputs_embeds[:, (start_idxs[s_idx]+3):start_idxs[s_idx+1]])
            #     input_embeds_with_qformer.append(inputs_embeds[:, (end_idxs[s_idx]):start_idxs[s_idx+1]])
            # # print("===================")
            # for xxx in input_embeds_with_qformer:
            #     print(xxx.shape)
            # input_embeds = torch.cat(input_embeds_with_qformer, dim=1)
            # print(input_embeds.shape)

            # inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            # print("only text is used !!")
            # print(inputs_embeds.shape)
            # inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            if short:
                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,     # if batch is 1, all 1 
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=max_len,
                    min_length=min_len,
                    length_penalty=length_penalty,
                )
            else:
                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,   # if batch is 1, all 1 
                    do_sample=False,
                    top_p=0.9,
                    temperature=1,
                    num_beams=num_beams,
                    max_new_tokens=30,
                    min_length=1,
                    repetition_penalty=1,
                    length_penalty=1,
                    num_return_sequences=1,
                )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        # if self._apply_lemmatizer:
        #     output_text = self._lemmatize(output_text)
        # print(text_input, output_text)
        return output_text

        # with self.maybe_autocast(dtype=torch.float32):
        #     # find 784 ("[") and 908 ("]")
        #     # print(input_tokens.input_ids)
        #     start_idxs = (input_tokens.input_ids[0]==784).nonzero(as_tuple=True)[0].tolist()
        #     end_idxs = (input_tokens.input_ids[0]==908).nonzero(as_tuple=True)[0].tolist()
        #     start_idxs = [int(s) for s in start_idxs]
        #     end_idxs = [int(s)+1 for s in end_idxs]
        #     # start_idxs = [0] + start_idxs
        #     print(start_idxs)
        #     print(end_idxs)
        #     inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)

        #     print(inputs_embeds.shape)
        #     start_idxs = start_idxs + [inputs_embeds.shape[1]+1]

        #     # for key in inputs_t5_objs.keys():
        #     #     print(inputs_t5_objs[key].shape)
        #     input_embeds_with_qformer = [inputs_embeds[:, :start_idxs[0]]]
        #     for s_idx in range(len(start_idxs)-1):
        #         obj_idx_now = obj_indices[s_idx]
        #         obj_query_now = inputs_t5_objs[obj_idx_now]

        #         input_embeds_with_qformer.append(obj_query_now)
        #         # input_embeds_with_qformer.append(inputs_embeds[:, (start_idxs[s_idx]+3):start_idxs[s_idx+1]])
        #         input_embeds_with_qformer.append(inputs_embeds[:, (end_idxs[s_idx]):start_idxs[s_idx+1]])
        #     # print("===================")
        #     for xxx in input_embeds_with_qformer:
        #         print(xxx.shape)
        #     input_embeds = torch.cat(input_embeds_with_qformer, dim=1)
        #     print(input_embeds.shape)

        #     # inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        #     if short:
        #         outputs = self.t5_model.generate(
        #             inputs_embeds=inputs_embeds,
        #             attention_mask=None,     # if batch is 1, all 1 
        #             do_sample=False,
        #             num_beams=num_beams,
        #             max_new_tokens=max_len,
        #             min_length=min_len,
        #             length_penalty=length_penalty,
        #         )
        #     else:
        #         outputs = self.t5_model.generate(
        #             inputs_embeds=inputs_embeds,
        #             attention_mask=None,   # if batch is 1, all 1 
        #             do_sample=False,
        #             top_p=0.9,
        #             temperature=1,
        #             num_beams=num_beams,
        #             max_new_tokens=30,
        #             min_length=1,
        #             repetition_penalty=1,
        #             length_penalty=1,
        #             num_return_sequences=1,
        #         )
        #     output_text = self.t5_tokenizer.batch_decode(
        #         outputs, skip_special_tokens=True
        #     )

        # # if self._apply_lemmatizer:
        # #     output_text = self._lemmatize(output_text)
        # # print(text_input, output_text)
        # return output_text

    def context_clear(self):
        self.contexts = None
        self.context_temp = None


    def predict_answers_interactive_multiturn(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        dense=False,
        short=False,
        **kwargs
    ):
        
        image = samples["image"]
        question = samples["question"]
        text_input = samples["text_input"]

        # print(image.shape)
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if text_input[0] is None:
            inputs_t5 = None
        else:
            text_Qformer = self.tokenizer(
                text_input,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            # Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                # attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output.last_hidden_state = query_output.last_hidden_state[:, :32]
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)

            # inputs_t5_objs[o_idx] = inputs_t5
            # atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.t5_tokenizer(
            samples["question"], padding="longest", return_tensors="pt"
        ).to(image.device)

        # encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.float32):
            
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            if self.contexts is None:
                if inputs_t5 is None:
                    inputs_embeds = torch.cat([inputs_embeds], dim=1)
                    self.context_temp = inputs_embeds
                else:
                    inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
                    self.contexts = inputs_t5
                    self.context_temp = inputs_embeds
            else:
                if inputs_t5 is None:
                    inputs_embeds = torch.cat([self.contexts, inputs_embeds], dim=1)
                    self.context_temp = inputs_embeds
                else:
                    self.contexts = self.context_temp
                    inputs_embeds = torch.cat([self.contexts, inputs_t5, inputs_embeds], dim=1)
                    self.context_temp = inputs_embeds
                    # self.contexts = inputs_embeds
            print("context dim", self.contexts.shape)
            
            if short:
                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,     # if batch is 1, all 1 
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=max_len,
                    min_length=min_len,
                    length_penalty=length_penalty,
                )
            else:
                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,   # if batch is 1, all 1 
                    do_sample=False,
                    top_p=0.9,
                    temperature=1,
                    num_beams=num_beams,
                    max_new_tokens=30,
                    min_length=1,
                    repetition_penalty=1,
                    length_penalty=1,
                    num_return_sequences=1,
                )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        # if self._apply_lemmatizer:
        #     output_text = self._lemmatize(output_text)
        # print(text_input, output_text)
        return output_text

    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        # max_txt_len  =16
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
