import json
import os
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only


import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import barrier, cleanup, get_rank
from threestudio.utils.ops import shifted_cosine_decay, shifted_expotional_decay
from threestudio.utils.typing import *


def hash_prompt(model: str, prompt: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}"
    return hashlib.md5(identifier.encode()).hexdigest()


@dataclass
class DirectionConfig:
    name: str
    prompt: Callable[[str], str]
    negative_prompt: Callable[[str], str]


@dataclass
class XLPromptProcessorOutput:
    text_embeddings: Float[Tensor, "N Nf"]
    uncond_text_embeddings: Float[Tensor, "N Nf"]

    def get_text_embeddings(
        self,
        batch_size
    ) -> Float[Tensor, "BB N Nf"]:
        text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
        uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
            batch_size, -1, -1
        )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)


class XLPromptProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        prompt: str = "a chinese style room"
        negative_prompt: str = ""
        
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        
        use_cache: bool = True
        spawn: bool = True

    cfg: Config

    @rank_zero_only
    def configure_text_encoder(self) -> None:
        raise NotImplementedError

    @rank_zero_only
    def destroy_text_encoder(self) -> None:
        raise NotImplementedError

    def configure(self) -> None:
        self._cache_dir = ".threestudio_cache/text_embeddings"  # FIXME: hard-coded path

        # view-dependent text embeddings
        
        with open(os.path.join("load/prompt_library.json"), "r") as f:
            self.prompt_library = json.load(f)
            
        # use provided prompt or find prompt in library
        self.prompt = self.preprocess_prompt(self.cfg.prompt)
        # use provided negative prompt
        self.negative_prompt = self.cfg.negative_prompt

        threestudio.info(
            f"Using prompt [{self.prompt}] and negative prompt [{self.negative_prompt}]"
        )

        self.prepare_text_embeddings()
        self.load_text_embeddings()

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        raise NotImplementedError

    @rank_zero_only
    def prepare_text_embeddings(self):
        os.makedirs(self._cache_dir, exist_ok=True)

        all_prompts = (
            [self.prompt]
            + [self.negative_prompt]
        )
        prompts_to_process = []
        for prompt in all_prompts:
            if self.cfg.use_cache:
                # some text embeddings are already in cache
                # do not process them
                cache_path = os.path.join(
                    self._cache_dir,
                    f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
                )
                if os.path.exists(cache_path):
                    threestudio.debug(
                        f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] are already in cache, skip processing."
                    )
                    continue
            prompts_to_process.append(prompt)

        if len(prompts_to_process) > 0:
            if self.cfg.spawn:
                ctx = mp.get_context("spawn")
                subprocess = ctx.Process(
                    target=self.spawn_func,
                    args=(
                        self.cfg.pretrained_model_name_or_path,
                        prompts_to_process,
                        self._cache_dir,
                    ),
                )
                subprocess.start()
                subprocess.join()
            else:
                self.spawn_func(
                    self.cfg.pretrained_model_name_or_path,
                    prompts_to_process,
                    self._cache_dir,
                )
            cleanup()

    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()
        self.text_embeddings = self.load_from_cache(self.prompt)[None, ...]
        self.uncond_text_embeddings = self.load_from_cache(self.negative_prompt)[
            None, ...
        ]
        threestudio.debug(f"Loaded text embeddings.")

    def load_from_cache(self, prompt):
        cache_path = os.path.join(
            self._cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        return torch.load(cache_path, map_location=self.device)

    def preprocess_prompt(self, prompt: str) -> str:
        if prompt.startswith("lib:"):
            # find matches in the library
            candidate = None
            keywords = prompt[4:].lower().split("_")
            for prompt in self.prompt_library["dreamfusion"]:
                if all([k in prompt.lower() for k in keywords]):
                    if candidate is not None:
                        raise ValueError(
                            f"Multiple prompts matched with keywords {keywords} in library"
                        )
                    candidate = prompt
            if candidate is None:
                raise ValueError(
                    f"Cannot find prompt with keywords {keywords} in library"
                )
            threestudio.info("Find matched prompt in library: " + candidate)
            return candidate
        else:
            return prompt

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B ..."], Float[Tensor, "B ..."]]:
        raise NotImplementedError


    def __call__(self) -> XLPromptProcessorOutput:
        return XLPromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
        )
