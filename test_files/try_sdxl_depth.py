import torch
import os
import numpy as np
from PIL import Image

from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionXLPipeline, StableDiffusionPipeline
from diffusers.utils import load_image
from torchvision.transforms import ToTensor
import torch._dynamo

torch._dynamo.config.suppress_errors = True
    

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
controlnet_pretrained_path = "/remote-home/hzp/Models/diffusers/controlnet-depth-sdxl-1.0"
sdxl_pretrained_path = "/remote-home/hzp/Models/stabilityai/stable-diffusion-xl-base-1.0"
vae_pretrained_path = "/remote-home/hzp/Models/madebyollin/sdxl-vae-fp16-fix"

def load_model():
    print("[INFO] Loading controlnet")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_pretrained_path,
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")

    print("[INFO] Loading vae")
    vae = AutoencoderKL.from_pretrained(vae_pretrained_path, torch_dtype=torch.float16).to("cuda")

    print("[INFO] Loading sdxl")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        sdxl_pretrained_path,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
    ).to("cuda")
    print("[INFO] Loaded sdxl")
    pipe.enable_model_cpu_offload()
    torch.cuda.empty_cache()
    return pipe

seed = 978364352
prompt = "A DSLR photo of partial view of a modern style living room, photorealistic"
controlnet_conditioning_scale = 0.55  # recommended for good generalization

# ! Using estimate depth to generate image
# rendered_dir = "test_files/rendered"
# depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
# feature_extractor = DPTImageProcessor.from_pretrained(pretrained_model_name_or_path="Intel/dpt-hybrid-midas")

# def get_depth_map(image):
#     image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
#     with torch.no_grad(), torch.autocast("cuda"):
#         depth_map = depth_estimator(image).predicted_depth

#     depth_map = torch.nn.functional.interpolate(
#         depth_map.unsqueeze(1),
#         size=(1024, 1024),
#         mode="bicubic",
#         align_corners=False,
#     )
#     depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
#     depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
#     depth_map = (depth_map - depth_min) / (depth_max - depth_min)
#     image = torch.cat([depth_map] * 3, dim=1)

#     image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
#     image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
#     return image


# estimate = True
# if estimate == True: 
#     estimate_dir = os.path.join(depth_dir.split('/')[0], "estimate")
#     if not os.path.exists(estimate_dir):
#         os.makedirs(estimate_dir, exist_ok=True)
#     save_dir = os.path.join(estimate_dir, controlnet_pretrained_path.split('/')[-1], f"{depth_dir.split('/')[-1]}_con-{controlnet_conditioning_scale}_seed-{seed}")

# render_img_list = os.listdir(rendered_dir)
# for img_name in render_img_list:
#     if '.png' in img_name:
#         # depth_image = Image.open(fp=os.path.join(depth_dir, img_name))
#         # depth_image = depth_image.resize((1024, 1024))
#         rendered_image = Image.open(fp=os.path.join(rendered_dir, img_name))
#         depth_image = get_depth_map(image=rendered_image)
#         depth_image.save(os.path.join(estimate_dir, f"{img_name[:2]}_estimate_depth.png")) #
        
        
#         generator = torch.Generator("cuda").manual_seed(seed)
#         images = pipe(
#             prompt, image=depth_image, num_inference_steps=30, controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator, return_dict=False, 
#             output_type="np"
#         )[0] # BHWC B=1 C=3 data_range: [0, 1] 
        
#         gen_image_PIL = Image.fromarray((images.squeeze() * 255.0).clip(0, 255).astype(np.uint8))
#         # gen_image_PIL.save("/remote-home/hzp/scene_generation/test.png")
#         gen_image_PIL.save( fp=os.path.join(save_dir, f"{img_name[:2]}_seed{seed}.png"))
#         print('done time')


# ! Using computed depth image to generate image
depth_dir = "/remote-home/hzp/scene_generation/test_files/depth-gray"
depth_img_list = os.listdir(depth_dir)
save_dir = os.path.join(depth_dir, controlnet_pretrained_path.split('/')[-1], f"con-{controlnet_conditioning_scale}_seed-{seed}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

pipe = load_model()
for img_name in depth_img_list:
    depth_image = Image.open(fp=os.path.join(depth_dir, img_name))
    generator = torch.Generator("cuda").manual_seed(seed)
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    images = pipe(
        prompt, image=depth_image, num_inference_steps=30, controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator, return_dict=False, 
        output_type="np"
    )[0] # BHWC B=1 C=3 data_range: [0, 1] 
    gen_image_PIL = Image.fromarray((images.squeeze() * 255.0).clip(0, 255).astype(np.uint8))
    gen_image_PIL.save(fp=os.path.join(save_dir, f"{img_name[:2]}.png"))
    

        


