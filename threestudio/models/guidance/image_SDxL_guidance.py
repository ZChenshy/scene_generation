from dataclasses import dataclass
import PIL
from PIL import Image
import numpy as np
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from threestudio.utils.saving import SaverMixin
from threestudio.utils.loss import l1_loss, ssim

@threestudio.register("stable-diffusion-xl-depth-controlnet-guidance-image")
class XLContrlnetGuidanceImage(BaseObject, SaverMixin):
    
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        controlnet_name_or_path: str = "diffusers/controlnet-depth-sdxl-1.0"
        vae_pretrained_path: str = "madebyollin/sdxl-vae-fp16-fix"
        depth_estimator_pretrained_path: str = "Intel/dpt-hybrid-midas"
        imgProcessor_pretrained_path: str = "Intel/dpt-hybrid-midas"
        
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        
        conditioning_scale: float = 0.55 # Diffusers recommends 0.5
        
        force_zeros_for_empty_prompt: bool = False 
        add_watermarker: bool = False 

        half_precision_weights: bool = False
        
        random_seed: int = 0
        num_inference_steps: int = 30 
        
    cfg : Config
    
    
    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion XL ControlNet and Depth Estimator...")
        
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        
        # Incase of decoding error occured when using float16
        if self.cfg.half_precision_weights:
            vae = AutoencoderKL.from_pretrained(
                self.cfg.vae_pretrained_path,
                torch_dtype=self.weights_dtype,
                cache_dir=self.cfg.cache_dir,
            ).to(self.device)
        else:
            vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
                subfolder="vae",
                cache_dir=self.cfg.cache_dir,
            ).to(self.device)
        
        controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_name_or_path,
            torch_dtype=self.weights_dtype, 
            cache_dir=self.cfg.cache_dir,
        ).to(self.device)
        
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            torch_dtype=self.weights_dtype,
            vae=vae,
            controlnet=controlnet,
            cache_dir=self.cfg.cache_dir,
        ).to(self.device)
        
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(
            pretrained_model_name_or_path=self.cfg.depth_estimator_pretrained_path,
            cache_dir=self.cfg.cache_dir,
        ).to(self.device)
        
        self.imgProcessor = DPTImageProcessor.from_pretrained(
            pretrained_model_name_or_path=self.cfg.imgProcessor_pretrained_path,
            cache_dir=self.cfg.cache_dir,
        ).to(self.device)
        cleanup()
        threestudio.info(f"Loaded Stable Diffusion XL ControlNet and Depth Estimator !")
        
        
    def get_depth_map(self, image) -> PIL.Image :
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image
    
    def __call__(
        self,
        rgb: Float[Tensor, "H W C"],
        prompt,
        prompt_embeds: Optional[torch.FloatTensor],
        negative_prompt_embeds: Optional[torch.FloatTensor],
        **kwargs,
    ):
        depth_PIL = self.get_depth_map(rgb)
        # if self.cfg.save_depth_map:
        #     self.save_grayscale_image(, "depth_map")
        
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.random_seed)
        
        # TODO: 
        # 1. Save to cache file incase of multiple calls (prompt, seed, condition_scale, num_inference_steps, camera_paras)
        # 2. Using prompt_embedding processed before
        
        gen_image = self.pipe(
            prompt, image=depth_PIL, num_inference_steps=self.cfg.num_inference_steps, controlnet_conditioning_scale=self.cfg.conditioning_scale, output_type="np",
            generator=generator, return_dict=False
        )[0] # BHWC B=1 C=3 data_range: [0, 1] np.array

        gen_image_PIL = Image.fromarray((gen_image.squeeze() * 255.0).clip(0, 255).astype(np.uint8))
        gen_image = torch.from_numpy(gen_image).squeeze(0).to(device=self.device, dtype=self.weights_dtype) # HWC data_range: [0, 1]
        
        
        Ll1_loss = l1_loss(gen_image, rgb)
        ssim_loss = ssim(gen_image, rgb)
        
        guidance_out = {
            "loss_l1": Ll1_loss,
            "loss_ssim": ssim_loss,
            "estimate_depth_PIL": depth_PIL,
            "gen_image_PIL": gen_image_PIL
        }
        return guidance_out