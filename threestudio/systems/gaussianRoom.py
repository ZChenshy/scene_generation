from dataclasses import dataclass, field
import torch
from tqdm import tqdm
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
import time
import numpy as np

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera

from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud

from scene_pc_utils import load_scene_pcd, save_ply

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from plyfile import PlyData, PlyElement

import io  
from PIL import Image  
import open3d as o3d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
@threestudio.register("gaussianroom-system")
class GaussianRoom(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        sh_degree: int = 0
        # TODO : 从参数中读取 load_path
        load_path: str = "./scene_pcd/LivingRoom-4719/scene_pcd.ply"  # scene point cloud path default
        
        
    cfg: Config
    
    def configure(self) -> None:
        self.radius = self.cfg.radius
        self.sh_degree =self.cfg.sh_degree
        self.load_path = self.cfg.load_path

        self.gaussian = GaussianModel(sh_degree = self.sh_degree)
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        
    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
    
    
    def forward(self, batch: Dict[str, Any],renderbackground = None) -> Dict[str, Any]: # 当使用self(batch)时，返回渲染结果
        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        self.viewspace_point_list = []
        for id in range(batch['c2w_3dgs'].shape[0]):
       
            viewpoint_cam  = Camera(c2w = batch['c2w_3dgs'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])
            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, depth, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["depth_3dgs"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            plt.imshow(image.cpu().detach().numpy())
            plt.axis('off') 
            plt.show()
        print()
        for id in range(batch['c2w_3dgs'].shape[0]):
            # 使用的c2w_3dgs生成，具体请查看uncond_out.py代码里的RandomCameraIterableDatasetCustom
            #请注意RandomCameraIterableDatasetCustom 的collated的返回值：'c2w_3dgs'对应的value
            viewpoint_cam  = Camera(c2w = batch['c2w_3dgs'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])

            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, depth, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["depth_3dgs"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)
            
            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii,self.radii)
                
            depth = render_pkg["depth_3dgs"]
            depth =  depth.permute(1, 2, 0)
            
            image =  image.permute(1, 2, 0)
            img = image.cpu().detach().numpy()
            img = (img * 255).astype(np.uint8)
            dep = depth.cpu().detach().numpy()
            # 如果图像是 RGB 格式，转换为 BGR 格式
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            path = "/remote-home/share/room_gen/room_render/livingroom"
            os.makedirs(path, exist_ok = True)
            stime = time.time()
            stime = str(stime).replace(".","_")
            cv2.imwrite(os.path.join(path,f"{stime}_render.jpg"), img)
            cv2.imwrite(os.path.join(path,f"{stime}_depth.jpg"), np.array(dep))
            # print("image has been saved")

            # plt.imshow(image.cpu().detach().numpy())
            # plt.axis('off') 
            # plt.show()
            images.append(image)
            depths.append(depth)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        self.visibility_filter = self.radii > 0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }
        
        
    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)
        
        if self.true_global_step > 500:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)

        self.gaussian.update_learning_rate(self.true_global_step)

        render_out = self.forward(batch) # batch 为相机参数
        # TODO : 在这里加入保存代码，将每次Gaussian渲染的结果保存，从而测试相机参数是否正确

        prompt_utils = self.prompt_processor() # TODO: 确定PromptProcessor的View-Dependent是否还需要
        images = render_out["comp_rgb"] # BHWC c=3
        depth_images = render_out["depth"] # BHWC c=1
        
        guidance_eval = (self.true_global_step % 200 == 0)
        # guidance_eval = False
        
        guidance_out = self.guidance(
            rgb=images, image_cond=depth_images, prompt_utils=prompt_utils, **batch, rgb_as_latents=False
            # guidance_eval=guidance_eval
        )

        loss = 0.0

        loss = loss + guidance_out['loss_sds'] * self.C(self.cfg.loss['lambda_sds'])
        
        loss_sparsity = (render_out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = render_out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        
        # if guidance_eval:
        #     self.guidance_evaluation_save(
        #         render_out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
        #         guidance_out["eval"],
        #     )
            
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}
    
    
    def on_before_optimizer_step(self, optimizer):

        with torch.no_grad():
            
            if self.true_global_step < 900: # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > 300 and self.true_global_step % 100 == 0: # 500 100
                    size_threshold = 20 if self.true_global_step > 500 else None # 3000
                    self.gaussian.densify_and_prune(0.0002 , 0.05, self.cameras_extent, size_threshold) 


    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if False else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch,testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )
            
    
    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last_3dgs.ply")
        self.gaussian.save_ply(save_path)
        
        # self.pointefig.savefig(self.get_save_path("pointe.png"))
        
        
        # o3d.io.write_point_cloud(self.get_save_path("shape.ply"), self.point_cloud)
        # self.save_gif_to_file(self.shapeimages, self.get_save_path("shape.gif"))
        save_ply(save_path, self.get_save_path(f"it{self.true_global_step}-test-color.ply"))
    
        
    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        
        opt = OptimizationParams(self.parser)
        point_cloud, self.cameras_extent = load_scene_pcd(self.load_path)
        
        self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)
        
        # * Test Initialization
        # self.gaussian.save_ply("./test.ply")
        
        self.pipe = PipelineParams(self.parser) 
        self.gaussian.training_setup(opt)
        
        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret
