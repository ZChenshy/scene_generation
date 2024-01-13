from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
import os

controlnet_pretrained_path = "/remote-home/share/Models/lllyasviel/control_v11f1p_sd15_depth"
sd_pretrain_path = "/remote-home/share/Models/runwayml/stable-diffusion-v1-5"
save_path = f"/remote-home/hzp/test/sd1.5_depth/{controlnet_pretrained_path.split('/')[-1]}"
os.makedirs(save_path, exist_ok=True)

seed = 978364352
conditioning_scale = 0.75

prompt = "A DSLR photo of a chinese style living room, viewed from ceil, photorealistic"
depth_image = Image.open("/remote-home/hzp/test/sdxl_depth/controlnet-depth-sdxl-1.0/A_DSLR_photo_of_a_chinese_style_living_room,_view_from_ceil,_photorealistic_depth.png")
depth_image = depth_image.resize((1024, 1024))
depth_image_array = np.array(depth_image)
print(depth_image_array.shape, depth_image_array.min(), depth_image_array.max())
 
depth_image.save(os.path.join(save_path, f"{prompt}_depth_seed{seed}_condition{conditioning_scale}_size{depth_image_array.shape[0]}.png"))
controlnet = ControlNetModel.from_pretrained(
    controlnet_pretrained_path, torch_dtype=torch.float16
)

generator = torch.manual_seed(seed)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_pretrain_path, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

image = pipe(prompt, depth_image, num_inference_steps=30, controlnet_conditioning_scale=conditioning_scale, generator=generator).images[0]

image.save(os.path.join(save_path, f"{prompt}_gen_seed{seed}_condition{conditioning_scale}_size{depth_image_array.shape[0]}.png"))
