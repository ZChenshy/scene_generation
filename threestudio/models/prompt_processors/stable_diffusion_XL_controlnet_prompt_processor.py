import json
import os
from dataclasses import dataclass
from regex import subf

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

import threestudio
from threestudio.models.prompt_processors.sdxl_base import XLPromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-XL-prompt-processor")
class StableDiffusionXLControlnetPromptProcessor(XLPromptProcessor):
    @dataclass
    class Config(XLPromptProcessor.Config):
        pass

    cfg: Config

    
    def configure_text_encoder(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer_2")
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.text_encoder = CLIPTextModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="text_encoder").to(self.device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="text_encoder_2").to(self.device)
        
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder_2.parameters():
            p.requires_grad_(False)
            
            
    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.tokenizer_2
        del self.text_encoder
        del self.text_encoder_2
        cleanup()
        
    
    def get_text_embeddings(
        self, prompt: Union[str, List[str]], 
        prompt_2: Optional[str],
        negative_prompt: Union[str, List[str]],
        negative_prompt_2: Optional[str],
        do_classifier_free_guidance: bool = True,
        force_zeros_for_empty_prompt: bool = True,
        
    ) -> Tuple[Float[Tensor, "B 77 2048"], Float[Tensor, "B 77 2048"], Float[Tensor, "B 2048"], Float[Tensor, "B 2048"]]:
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        
        prompt_2 = prompt_2 or prompt
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            
            text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    threestudio.info(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )
                    
            prompt_embeds = text_encoder(
                    text_input_ids.to(self.device),
                    output_hidden_states=True,
                )
            
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)
            
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1) # [1, 77, 768 + 1280 = 2048] 
        
        zero_out_negative_prompt = negative_prompt is None and force_zeros_for_empty_prompt
        
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt: # Negative Prompt 输出的Embedding为全0
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            
        uncond_tokens: List[str]
        uncond_tokens = [negative_prompt, negative_prompt_2]
            
        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(self.device),
                output_hidden_states=True,
            )
            
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
        
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=self.device)
        
        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=self.device)
        
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    '''
        Shape:
            prompt_embeds: [1, 77, 2048]
            negative_prompt_embeds: [1, 77, 2048]
            pooled_prompt_embeds: [1, 2048]
            negative_pooled_prompt_embeds: [1, 2048]
    '''
    
if __name__ == '__main__':
    print('test')