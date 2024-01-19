from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.SDXL_base import XLPromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

# TODO: 不再使用PipeLine，直接调用Unet和VAE，进行Encode和Noise Predict
@threestudio.register("stable-diffusion-xl-depth-controlnet-guidance")
class XLContrlnetGuidance(BaseObject):
    
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        controlnet_name_or_path: str = "diffusers/controlnet-depth-sdxl-1.0"
        
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        
        guidance_scale: float = 100 # classifier free guidance
        condition_scale: float = 0.75 # Diffusers recommends 0.5
        
        force_zeros_for_empty_prompt: bool = False # TODO 实现并测试这个参数对效果的影响
        add_watermarker: bool = False 

        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = False
        
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        
        diffusion_steps: int = 30 # 与ThreeStudio中ControlNet Guidance相同默认为20
        
        weighting_strategy: str = "sds"
        
    cfg : Config
    
    
    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion XL ControlNet ...")

        self.weights_dtype = (
            torch.bfloat16 if self.cfg.half_precision_weights else torch.float32
        )
        
        self.controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_name_or_path,
            torch_dtype=self.weights_dtype, 
            cache_dir=self.cfg.cache_dir,
        ).to(self.device)
        
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            torch_dtype=self.weights_dtype,
            controlnet=controlnet,
            cache_dir=self.cfg.cache_dir,
        ).to(self.device)
        
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        
        # TODO: determine whether to use these options
        # if self.cfg.enable_memory_efficient_attention:
        #     if parse_version(torch.__version__) >= parse_version("2"):
        #         threestudio.info(
        #             "PyTorch2.0 uses memory efficient attention by default."
        #         )
        #     elif not is_xformers_available():
        #         threestudio.warn(
        #             "xformers is not available, memory efficient attention is not enabled."
        #         )
        #     else:
        #         self.pipe.enable_xformers_memory_efficient_attention()
        
        # if self.cfg.enable_sequential_cpu_offload:
        #     self.pipe.enable_sequential_cpu_offload()
            
        # if self.cfg.enable_attention_slicing:
        #     self.pipe.enable_attention_slicing(1)

        # if self.cfg.enable_channels_last_format:
        #     self.pipe.unet.to(memory_format=torch.channels_last)
            
        cleanup()
            
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

        threestudio.info(f"Loaded Stable Diffusion XL ControlNet!")
    
    
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
    
    
    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._get_add_time_ids
    def _get_add_time_ids(
        self, 
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0), 
        target_size: Tuple[int, int] = None,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)   
        add_time_ids = torch.tensor([add_time_ids], dtype=self.weights_dtype) # dtype should be the same as prompt
        return add_time_ids
    
    
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
        latent: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        controlnet_cond: Float[Tensor, "..."],
        conditioning_scale: float,
        added_cond_kwargs,
    ) -> Float[Tensor, "..."]:
        return self.controlnet(
            latent.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            controlnet_cond=controlnet_cond.to(self.weights_dtype),
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        
        
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 1024 1024"]
    ) -> Float[Tensor, "B 4 128 128"]:
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
        added_cond_kwargs,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
        ).sample.to(input_dtype)
    
    
    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 128 128"],
        latent_height: int = 128,
        latent_width: int = 128,
    ) -> Float[Tensor, "B 3 1024 1024"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        image = image.clamp(0, 1)
        return image.to(input_dtype)
    
    
    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 2048"],
        add_text_embeds: Float[Tensor, "BB 1280"],
        add_time_ids: Int[Tensor, "B 6"],
        latents: Float[Tensor, "B 4 128 128"],
        t: Int[Tensor, "B"],
        image_cond: Float[Tensor, "B 3 1024 1024"],
    ):
        with torch.no_grad():
            image_cond = torch.cat([image_cond] * 2)

            noise = torch.rand_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            
            add_text_embeds.to(self.device, self.weights_dtype)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                    latent=latent_model_input,
                    t=torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=image_cond,
                    conditioning_scale=self.cfg.condition_scale,
                    added_cond_kwargs=added_cond_kwargs,
                )
            
            noise_pred = self.forward_control_unet(
                latents=latent_model_input,
                t=torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
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
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")
        
        
    
    
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        image_cond: Float[Tensor, "B H W C"],
        prompt_utils: XLPromptProcessorOutput,
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0] 
        assert batch_size > 0
        if not rgb_as_latents:
            assert (rgb.shape[1], rgb.shape[2]) == (image_cond.shape[1], image_cond.shape[2])
        assert len(rgb.shape) == len(image_cond.shape) == 4
        
        rgb = torch.clamp(rgb, 0, 1)
        rgb_BCHW = rgb.permute(0, 3, 1, 2) # B H W C -> B C H W
        latents: Float[Tensor, "B 4 128 128"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW,
                (128, 128),
                mode="bilinear",
                align_corners=False,
            )
        else:
            rgb_BCHW_1024 = F.interpolate(
                rgb_BCHW,
                (1024, 1024),
                mode="bilinear",
                align_corners=False,
            )
            latents = self.encode_images(rgb_BCHW_1024)
            
        depth_BCHW = self.normalized_image(image_cond)
        if depth_BCHW.shape[1] == 1:
            depth_BCHW = torch.cat([depth_BCHW] * 3, dim=1)
        elif depth_BCHW.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Depth image has {depth_BCHW.shape[1]} channels, expected 1 or 3.")
        
        depth_BCHW_1024 = F.interpolate(
            depth_BCHW, 
            size=(1024, 1024), 
            mode="bilinear", 
            align_corners=False
        )
        
        text_embeddings = prompt_utils.get_text_embeddings(batch_size) # BB 77 2048, cond, uncond
        add_text_embeds = prompt_utils.get_pooled_prompt_embeds(batch_size) # BB 1280, cond, uncond
        
        # TODO: Fix hard coded value
        add_time_ids = self._get_add_time_ids(
            original_size=(1024, 1024),
            crops_coords_top_left=(0, 0), 
            target_size=(1024, 1024),
        ).repeat(batch_size, 1) # [B, 6]
        
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        
        grad = self.compute_grad_sds(
            text_embeddings=text_embeddings,
            add_text_embeds=add_text_embeds,
            add_time_ids=add_time_ids,
            latents=latents,
            t=t,
            image_cond=depth_BCHW_1024,
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
