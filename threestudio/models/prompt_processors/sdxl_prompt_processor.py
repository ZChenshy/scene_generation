import os
from dataclasses import dataclass

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import threestudio
from threestudio.models.prompt_processors.SDXL_base import XLPromptProcessor, hash_prompt, hash_pool_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-xl-prompt-processor")
class StableDiffusionXLPromptProcessor(XLPromptProcessor):
    @dataclass
    class Config(XLPromptProcessor.Config):
        pass
    
    cfg: Config

     ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder_2"
        ).to(self.device)

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
    ###
    

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir, cache_dir_pool):
        '''
            prompts: list of strings list, [(0:prompt, 1:prompt_2)] || [(0:negative_prompt, 1:negative_prompt_2)]
        '''
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="tokenizer_2"
        )
        tokenizers = [tokenizer, tokenizer_2]
        
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="text_encoder",
            device_map="auto"
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="text_encoder_2",
            device_map="auto"
        )
        text_encoders = [text_encoder, text_encoder_2]
        
        with torch.no_grad():
            for i in range(len(prompts)):
                prompt_embeds_list = []
                for prompt, tokenizer, text_encoder in zip(prompts[i], tokenizers, text_encoders):
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
                        threestudio.warn(
                            "The following part of your input was truncated because CLIP can only handle sequences up to"
                            f" {tokenizer.model_max_length} tokens: {removed_text}"
                        )
                    
                    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True)
                    
                    pooled_prompt_embeds = prompt_embeds[0]
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                    
                    prompt_embeds_list.append(prompt_embeds)
                    
                prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
                
                torch.save(
                    prompt_embeds, 
                    os.path.join(
                        cache_dir, 
                        f"{hash_prompt(pretrained_model_name_or_path, prompts[i][0], prompts[i][1])}.pt"
                    )
                )
            
                # * Only save pooled prompt embeds gained from text_encoder_2
                torch.save(
                    pooled_prompt_embeds, 
                    os.path.join(
                        cache_dir_pool, 
                        f"{hash_pool_prompt(pretrained_model_name_or_path, prompts[i][0], prompts[i][1])}.pt"
                    )
                )
            
        del text_encoder
        del text_encoder_2