from ast import Not
from dataclasses import dataclass, field

# <<< for Debugging >>>
import cv2
import os
import numpy as np
# <<< for Debugging >>

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("depth-controlnet-SD1.5-guidance")
class ControlnetGuidance(BaseObject):
    
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        controlnet_name_or_path: str = "lllyasviel/control_v11f1p_sd15_depth" 
        ddim_scheduler_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        
        guidance_scale: float = 7.5 # classifier free guidance
        condition_scale: float = 0.75 # Diffusers recommends 0.5
        
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        fixed_size: int = -1 # TODO: Determine what this is for
        
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        
        diffusion_steps: int = 20 # 与ThreeStudio中ControlNet Guidance相同默认为20
        
        weighting_strategy: str = "sds"

    cfg : Config
    
    
    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion v1.5 Depth ControlNet ...")
        
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }
        
        controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_name_or_path, torch_dtype=self.weights_dtype, cache_dir=self.cfg.cache_dir,
        )
        
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, controlnet=controlnet, **pipe_kwargs
        ).to(self.device)
        
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        
        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
        
        del self.pipe.text_encoder
        cleanup()
        
        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.controlnet = self.pipe.controlnet.eval()
        
        
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.controlnet.parameters():
            p.requires_grad_(False)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value
        
        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        
        self.grad_clip_val: Optional[float] = None
        
        threestudio.info(f"Loaded Stable Diffusion v1.5 Depth ControlNet !")


    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        
        
    # TODO: 使得输入的深度图像的像素值范围为[0, 1]，同时使近距离的区域像素值更大，远距离的区域像素值更小（与Gaussian渲染的深度进行结合）
    def normalized_image(self, image: Float[Tensor, "B H W C"]) -> Float[Tensor, "B C H W"]:
        image = image.permute(0, 3, 1, 2)
        min_values = torch.amin(image, dim=[2, 3], keepdim=True)
        max_values = torch.amax(image, dim=[2, 3], keepdim=True)
        normalized_image = (image - min_values) / (max_values - min_values)
        normalized_image.to(image.dtype)
        return normalized_image
    
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_controlnet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        image_cond: Float[Tensor, "..."],
        condition_scale: float,
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        '''
            latents: BB 4 64 64
            t: single value
            image_cond: BB 3 512 512 [0, 1]
            condition_scale: single value
            encoder_hidden_states: BB 77 768 cond, uncond
        '''
        return self.controlnet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=image_cond.to(self.weights_dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )
    
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)
    
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_control_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs,
        down_block_additional_residuals,
        mid_block_additional_residual,
    ) -> Float[Tensor, "..."]:
        '''
            latents: BB 4 64 64
            t: single value
            encoder_hidden_states: BB 77 768 
            cond uncond: 该顺序对应noise_pred_text, noise_pred_uncond的顺序,可以改变顺序
        '''
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)


    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)


    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"], # 在__call__中，从prompt_utils中获取，加入ViewDependtenPromptProcessor之后再做修改
        latents: Float[Tensor, "B 4 64 64"],
        image_cond: Float[Tensor, "B 3 H W"], # 深度图的通道为1,但ControlNet输入要求为3，因此先前已经进行了拼接
        t: Int[Tensor, "B"], 
    ):
        with torch.no_grad():
            image_cond = torch.cat([image_cond] * 2) # BB 3 H W
            
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                image_cond=image_cond,
                condition_scale=self.cfg.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # perform classifier-free guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)

        if self.cfg.weighting_strategy == "sds":
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            ) 
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            grad = w * (noise_pred - noise)
            return grad
        elif self.cfg.weighting_strategy == "csd":
            noise_pred = self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            grad = w * (noise_pred)
            return grad
        else:
            raise NotImplementedError

    
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        image_cond: Float[Tensor, "B H W C"], # TODO: 对Gaussian渲染的深度图进行预处理
        prompt_utils: PromptProcessorOutput,
        rgb_as_latents=False,
        # elevation: Float[Tensor, "B"],
        # azimuth: Float[Tensor, "B"],
        # camera_distances: Float[Tensor, "B"],
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        if not rgb_as_latents:
            assert rgb.shape[1] == image_cond.shape[1] # 让渲染图与条件图大小相同
            assert len(rgb.shape) == len(image_cond.shape) == 4
        
        # TODO: 使用ViewDependtent Prompt Processor之后，需要修改
        temp = torch.zeros(1).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False) # BB 77 768, cond, uncond
        
        depth_img_BCHW = self.normalized_image(image_cond) 
        
        if depth_img_BCHW.shape[1] == 1:
            depth_img_BCHW = torch.cat([depth_img_BCHW, depth_img_BCHW, depth_img_BCHW], dim=1) # 1C -> 3C
        elif depth_img_BCHW.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Depth image channel should be 1 or 3, but got {depth_img_BCHW.shape[1]}")
            
        depth_img_BCHW_512 = F.interpolate(
            depth_img_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        
        rgb_BCHW = rgb.permute(0, 3, 1, 2) # B H W C -> B C H W
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW = torch.clamp(rgb_BCHW, 0, 1) # Gaussian渲染图片像素值会有偏差，进行Clamp
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            ) # 为了匹配Encoder的尺寸 
            latents: Float[Tensor, "B 4 64 64"]
            latents = self.encode_images(rgb_BCHW_512)
        
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        grad = self.compute_grad_sds(
            text_embeddings=text_embeddings, latents=latents, image_cond=depth_img_BCHW_512, t=t
        )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        
        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }
        
        return guidance_out


    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
        