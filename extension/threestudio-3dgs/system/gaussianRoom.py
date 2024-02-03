from dataclasses import dataclass
import os
import numpy as np
from sympy import false
import torch
from PIL import Image
from random import choice
import torchvision.transforms as transforms
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer
from threestudio.utils.loss import ssim, tv_loss, l1_loss
from threestudio.utils.typing import *
from ..geometry.gaussian_base import BasicPointCloud


@threestudio.register("gaussianRoom-system")
class GaussianRoom(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # TODO: 写在配置文件中
        visualize_samples: bool = False
        # cam_path: str = "coarse_room/camera_config/camsInfo.pkl" 
        cam_path: str = "coarse_room/camera_config/panocamsInfo.pkl"
        single_iter: int = 100 # 每个角度迭代的次数 # TODO: 写在配置文件中
        
    cfg: Config
    
    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        
    def on_fit_start(self) -> None:
        super().on_fit_start()
    
    
    def configure_optimizers(self):
        optim = self.geometry.optimizer
        
        if hasattr(self, "merged_optimizer"):
            return [optim]
        
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]: # 当使用self(batch)时，返回渲染结果
        self.geometry.update_learning_rate(self.global_step)
        outputs = self.renderer.batch_forward(batch)
        return outputs
        
        
    def on_fit_start(self) -> None:
        super().on_fit_start()
        save_path = self.get_save_path(f"point_cloud_it{self.global_step}.ply")
        self.geometry.save_ply(save_path)
        
        
    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        
        # 每个视角的最多迭代次数
            
        
        out = self.forward(batch) # HWC

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        # out["comp_rgb"]  # BHWC, c=3, [0, 1]
        # out["comp_depth"]# BHWC, c=1, absolute depth, not normalized
        viewspace_point_tensor = out["viewspace_points"]
        
        cam_idx = batch["cam_idx"]
        if not os.path.exists(self.get_save_path(f"rendered/{cam_idx}-rgb.png")):
            self.save_rgb_image(
                f"rendered/{cam_idx}-rgb.png",
                img=out["comp_rgb"].squeeze(),
                data_format="HWC",
            )
        
            self.save_colorized_depth(
                filename=f"depth-gray/{cam_idx}-depth.png", 
                depth=out["comp_depth"].squeeze(),
                cmap=None, # Jet is colorized, None is grayscale
            )
            
            self.save_colorized_depth(
                filename=f"depth-jet/{cam_idx}-depth.png",
                depth=out["comp_depth"].squeeze(),
                cmap="jet", # Jet is colorized, None is grayscale
            )
            
        guidance_inp = out["comp_rgb"] # BHWC, c=3, [0, 1] # 利用渲染图片做深度预测
        guidance_cond = Image.open(self.get_save_path(f"depth-gray/{cam_idx}-depth.png")) # 利用计算的深度做Condition
        # regen = True if not os.path.exists(self.get_save_path(f"generated/{batch["cam_idx"]}-rgb.png")) else False # For image Guidance
        
        
        # For Image guidance
        # generated_image = None
        #
        # if not regen:
        #     generated_image = Image.open(self.get_save_path(f"generated/{viewpoint_cam_idx}-rgb.png")) 
        # 
        # guidance_out = self.guidance(
        #     rgb=guidance_inp,
        #     condition_image=guidance_cond, 
        #     prompt="A DSLR photo of a modern style livingroom, partial view",
        #     regen=regen,
        #     generated_image=generated_image,
        # )
        #
        # if regen: 
        #     self.save_PIL(
        #         filename=f"generated/{viewpoint_cam_idx}-rgb.png", 
        #         pil_image=guidance_out["gen_image_PIL"],
        #     )
            
            # self.save_PIL(
            #     filename=f"generated/{viewpoint_cam_idx}-depth.png", 
            #     pil_image=guidance_out["estimate_depth_PIL"],
            # )
        
        
        # SDS Guidance
        # input: BHWC, c=3, [0, 1]
        # condition: PIL.Image
        guidance_out = self.guidance(
            guidance_inp, guidance_cond, self.prompt_utils, **batch, rgb_as_latents=False
        )
        
        loss_img = 0.0
        loss = 0.0
        
        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                loss_img += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                ) # 包括l1 SSIM
                self.log(f"train/{name}", value)
                
                
        xyz_mean = None
        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            loss_position = xyz_mean.mean()
            self.log(f"train/loss_position", loss_position)
            loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

        if self.cfg.loss["lambda_opacity"] > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).sum()
            self.log(f"train/loss_opacity", loss_opacity)
            loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_scales"] > 0.0:
            scale_sum = torch.sum(self.geometry.get_scaling)
            self.log(f"train/scales", scale_sum)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        if self.cfg.loss["lambda_tv_loss"] > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                out["comp_rgb"].permute(0, 3, 1, 2)
            )
            self.log(f"train/loss_tv", loss_tv)
            loss += loss_tv
            
        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
        ):
            loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
                # tv_loss(out["comp_normal"].permute(0, 3, 1, 2)) # ! 这里的comp_normal是怎么得到的
                tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            )
            self.log(f"train/loss_depth_tv", loss_depth_tv)
            loss += loss_depth_tv

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
            
            
        loss_img.backward(retain_graph=True) # SDS Or ImageGuidance Loss
        loss.backward(retain_graph=True)
        
        iteration = self.global_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
        
        if loss > 0:
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": loss_img} # SDS Or Image Guidace Loss
            
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0]}.png",
            [
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
            step=self.global_step,
        )
        # 每次Evaluation都保存点云
        save_path = self.get_save_path(f"point_cloud_it{self.global_step}.ply")
        self.geometry.save_ply(save_path)    
    
    def on_validation_epoch_end(self):
        pass
    
    
    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0]}.png",
            [
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
        )
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)
            
            
    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.global_step,
        )
        
    
    def on_load_checkpoint(self, ckpt_dict) -> None:
        num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        super().on_load_checkpoint(ckpt_dict)
        
    