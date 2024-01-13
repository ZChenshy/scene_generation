import torch
import os
import numpy as np
from PIL import Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionXLPipeline, StableDiffusionPipeline
from diffusers.utils import load_image
from torchvision.transforms import ToTensor
controlnet_pretrained_path = "/remote-home/share/room_gen/Models/diffusers/controlnet-depth-sdxl-1.0"
sdxl_pretrained_path = "/remote-home/share/room_gen/Models/stabilityai/stable-diffusion-xl-base-0.9"
vae_pretrained_path = "/remote-home/share/room_gen/Models/madebyollin/sdxl-vae-fp16-fix"
save_path = f"/remote-home/hzp/test/sdxl_0.9_depth/{controlnet_pretrained_path.split('/')[-1]}"
os.makedirs(save_path, exist_ok=True)


depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
    controlnet_pretrained_path,
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained(vae_pretrained_path, torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    sdxl_pretrained_path,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_model_cpu_offload()

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

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


seed = 978364352
prompt = "stormtrooper lecture, photorealistic"
# prompt = "A DSLR photo of a chinese style living room, photorealistic"
image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
controlnet_conditioning_scale = 0.55  # recommended for good generalization

# depth_image = Image.open("/remote-home/hzp/test/sdxl_depth/controlnet-depth-sdxl-1.0-mid/A_DSLR_photo_of_a_chinese_style_living_room,_floor_view_depth.png")
depth_image = get_depth_map(image)

depth_image = depth_image.resize((1024, 1024))
depth_image.save(os.path.join(save_path, f"{prompt.replace(' ', '_')}_depth.png"))

depth_np = np.array(depth_image)
print(depth_np.shape, depth_np.min(), depth_np.max())
generator = torch.Generator("cuda").manual_seed(seed)
images = pipe(
    prompt, image=depth_image, num_inference_steps=30, controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator
).images
images[0]

images[0].save(os.path.join(save_path, f"{prompt.replace(' ', '_')}_sdxl_gen_con{controlnet_conditioning_scale}_seed{seed}.png"))
