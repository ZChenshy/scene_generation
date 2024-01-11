import os
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import barrier, cleanup
from threestudio.utils.typing import *


def hash_prompt(model: str, prompt: str, prompt2: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}-{prompt2}"
    return hashlib.md5(identifier.encode()).hexdigest()

def hash_pool_prompt(model: str, prompt: str, prompt2: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}-{prompt2}-pool"
    return hashlib.md5(identifier.encode()).hexdigest()

# TODO: Using for view dependent prompt
# @dataclass
# class DirectionConfig:
#     name: str
#     prompt: Callable[[str], str]
#     negative_prompt: Callable[[str], str]
#     condition: Callable[
#         [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
#         Float[Tensor, "B"],
#     ]


@dataclass
class XLPromptProcessorOutput:
    text_embeddings: Float[Tensor, "batch_size 77 2048"]
    uncond_text_embeddings: Float[Tensor, "batch_size 77 2048"]
    pooled_prompt_embeds: Float[Tensor, "batch_size 2048"]
    negative_pooled_prompt_embeds: Float[Tensor, "batch_size 2048"]

    def get_text_embeddings(
        self,
        # elevation: Float[Tensor, "B"],
        # azimuth: Float[Tensor, "B"],
        # camera_distances: Float[Tensor, "B"],
        batch_size: int,
        view_dependent_prompting: bool = False,
    ) -> Float[Tensor, "BB N Nf"]:
        
        if view_dependent_prompting:
            raise NotImplementedError
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)
    
    def get_pooled_prompt_embeds(
        self,
        batch_size: int,
        view_dependent_prompting: bool = False
    ) -> Float[Tensor, "BB NF"]:
        if view_dependent_prompting:
            raise NotImplementedError
        else:
            pooled_prompt_embeds = self.pooled_prompt_embeds.expand(batch_size, -1)
            negative_pooled_prompt_embeds = self.negative_pooled_prompt_embeds.expand(batch_size, -1)
        return torch.cat([pooled_prompt_embeds, negative_pooled_prompt_embeds], dim=0)


class XLPromptProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        prompt: str = "A DSLR photo of a chinese style living room"
        prompt_2: Optional[str] = None
        
        negative_prompt: str = ""
        negative_prompt_2: Optional[str] = None
        
        use_cache: bool = True
        spawn: bool = True
        
        # 当negative_prompt 为空时，是否使用全零的negative_prompt_embeddings
        # force_zeros_for_empty_prompt: bool = True
        
        pretrained_model_name_or_path: str = "/remote-home/share/Models/stabilityai/stable-diffusion-xl-base-1.0"
    
    cfg: Config
    
    
    @rank_zero_only
    def configure_text_encoder(self) -> None:
        raise NotImplementedError
    
    
    @rank_zero_only
    def destroy_text_encoder(self) -> None:
        raise NotImplementedError
    
    
    def configure(self) -> None:
        self._cache_dir = ".threestudio_cache/XL/text_embeddings"  # FIXME: hard-coded path
        self._cache_dir_pool = ".threestudio_cache/XL/text_embeddings_pool"  # FIXME: hard-coded path
        
        # use provided prompt or find prompt in library
        self.prompt = self.cfg.prompt
        self.prompt_2 = self.cfg.prompt_2 or self.prompt
        
        # use provided negative prompt
        self.negative_prompt = self.cfg.negative_prompt or ""
        self.negative_prompt_2 = self.cfg.negative_prompt_2 or self.negative_prompt
        
        threestudio.info(
            f"Using prompt:\n prompt_1: [{self.prompt}], \n prompt_2: [{self.prompt_2}]; \n negative_prompt1: [{self.negative_prompt}], \n negative_prompt2: [{self.negative_prompt_2}]"
        )
        self.prepare_text_embeddings()
        self.load_text_embeddings()
        
        
    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir, cache_dir_pool):
        raise NotImplementedError
    
    
    @rank_zero_only
    def prepare_text_embeddings(self) -> None:
        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._cache_dir_pool, exist_ok=True)
        
        prompts_to_process = []
        if self.cfg.use_cache:
            cache_path = os.path.join(
                self._cache_dir, hash_prompt(self.cfg.pretrained_model_name_or_path, self.prompt, self.prompt_2)
            )
            cache_path_pool = os.path.join(
                self._cache_dir_pool, hash_pool_prompt(self.cfg.pretrained_model_name_or_path, self.prompt, self.prompt_2)
            )
            
            if os.path.exists(cache_path):
                threestudio.debug(
                    f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{self.prompt}, {self.prompt_2}] are already in cache, skip processing."
                )
            if os.path.exists(cache_path_pool):
                threestudio.debug(
                    f"Text pool embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{self.prompt}, {self.prompt_2}] are already in cache, skip processing."
                )
                
        prompts_to_process.append([self.prompt, self.prompt_2])
            
        if self.cfg.use_cache:
            cache_path = os.path.join(
                self._cache_dir,
                hash_prompt(self.cfg.pretrained_model_name_or_path, self.negative_prompt, self.negative_prompt_2),
            )
            cache_path_pool = os.path.join(
                self._cache_dir_pool,
                hash_pool_prompt(self.cfg.pretrained_model_name_or_path, self.negative_prompt, self.negative_prompt_2),
            )
            
            if os.path.exists(cache_path):
                threestudio.debug(
                    f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and negative prompt [{self.negative_prompt}, {self.negative_prompt_2}] are already in cache, skip processing."
                )
            if os.path.exists(cache_path_pool):
                threestudio.debug(
                    f"Text pool embeddings for model {self.cfg.pretrained_model_name_or_path} and negative prompt [{self.negative_prompt}, {self.negative_prompt_2}] are already in cache, skip processing."
                )
                
        prompts_to_process.append([self.negative_prompt, self.negative_prompt_2])
            
        if len(prompts_to_process) > 0:
            if self.cfg.spawn:
                ctx = mp.get_context("spawn")
                subprocess = ctx.Process(
                    target=self.spawn_func,
                    args=(
                        self.cfg.pretrained_model_name_or_path,
                        prompts_to_process,
                        self._cache_dir,
                        self._cache_dir_pool,
                    ),
                )
                subprocess.start()
                subprocess.join()
            else:
                self.spawn_func(
                    self.cfg.pretrained_model_name_or_path,
                    prompts_to_process,
                    self._cache_dir,
                    self._cache_dir_pool,
                )
            cleanup()
    
    
    def load_text_embeddings(self):
        barrier()
        self.text_embeddings = self.load_from_cache(self.prompt, self.prompt_2)
        self.uncond_text_embeddings = self.load_from_cache(self.negative_prompt, self.negative_prompt_2)
        self.pooled_prompt_embeds = self.load_from_cache(self.prompt, self.prompt_2, pooled=True)
        self.negative_pooled_prompt_embeds = self.load_from_cache(self.negative_prompt, self.negative_prompt_2, pooled=True)
        threestudio.debug(f"Loaded text embeddings.")
    
    
    def load_from_cache(self, prompt, prompt2, pooled: bool=False):
        if not pooled:
            cache_path = os.path.join(
                self._cache_dir,
                f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt, prompt2)}.pt",
            )
        else:
            cache_path = os.path.join(
                self._cache_dir_pool,
                f"{hash_pool_prompt(self.cfg.pretrained_model_name_or_path, prompt, prompt2)}.pt",
            )
            
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}], prompt_2[{prompt2}] not found."
            )
        return torch.load(cache_path, map_location=self.device)
    
    
    def get_text_embeddings(
        self, 
        prompt: Union[str, List[str]], 
        prompt2: Optional[str], 
        negative_prompt: Union[str, List[str]], 
        negative_prompt2: Optional[str]
    ) -> Tuple[Float[Tensor, "B ..."], Float[Tensor, "B ..."]]:
        raise NotImplementedError
        

    def __call__(self) -> XLPromptProcessorOutput:
        return XLPromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
            pooled_prompt_embeds=self.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds
        )