from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from random import randint
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
        cam_path: str = "coarse_room/camera_config/camsInfo.pkl" 
        
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
        self.transform = transforms.ToTensor()
        
        
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
        viewpoint_cam_idx = (randint(0, len(self.cam_list)-1))
        
        viewpoint_cam = {
            "c2w": self.cam_list["c2w"][viewpoint_cam_idx].to(self.device),
            "w2c": self.cam_list["w2c"][viewpoint_cam_idx].to(self.device),
            "fovx": self.cam_list["fovx"][viewpoint_cam_idx],
            "fovy": self.cam_list["fovy"][viewpoint_cam_idx],
            "width": self.cam_list["width"],
            "height": self.cam_list["height"]
        }
        
        out = self.forward(viewpoint_cam) # HWC

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]  # BHWC, c=3, [0, 1]
        guidance_cond = out["comp_depth"]# BHWC, c=1, absolute depth, not normalized
        viewspace_point_tensor = out["viewspace_points"]
    

        self.save_rgb_image(
            f"rendered/{self.true_global_step}-rgb.png",
            img=guidance_inp,
            data_format="HWC",
        )
        
        self.save_colorized_depth(
            f"depth/{self.true_global_step}-depth.png", 
            guidance_cond,
        )
        
        # * SDS >>>
        # guidance_out = self.guidance(
        #     rgb=guidance_inp, image_cond=guidance_cond, 
        #     prompt_utils=self.prompt_utils, 
        #/     **batch, rgb_as_latents=False
        # )
        guidance_inp = guidance_inp.permute(2, 0, 1) # CHW c=3 [0, 1]
        gt_image = Image.open("/remote-home/hzp/scene_generation/outputs/gaussiandroom-sd/test@20240126-053353/save/controlnet-depth-sdxl-1.0/estimateDepth_con-0.55_seed-978364352/7-_seed978364352.png").resize((512, 512))
        gt_image = self.transform(gt_image).to(self.device) # CHW c=3 [0, 1]
        
        L1_loss = l1_loss(guidance_inp, gt_image)
        loss = (1.0 - 0.2) * L1_loss + 0.2 * (1.0 - ssim(guidance_inp, gt_image))
        
        # loss_sds = 0.0
        # loss = 0.0
        
        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss_l1 += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                ) # 包括l1 SSIM
            elif name == "estimate_depth_PIL":
                value.save(self.get_save_path(f"{name}/{self.true_global_step}.png"))
            elif name == "gen_image_PIL":
                value.save(self.get_save_path(f"{name}/{self.true_global_step}.png"))
                
                
        # xyz_mean = None
        # if self.cfg.loss["lambda_position"] > 0.0:
        #     xyz_mean = self.geometry.get_xyz.norm(dim=-1)
        #     loss_position = xyz_mean.mean()
        #     self.log(f"train/loss_position", loss_position)
        #     loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

        # if self.cfg.loss["lambda_opacity"] > 0.0:
        #     scaling = self.geometry.get_scaling.norm(dim=-1)
        #     loss_opacity = (
        #         scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
        #     ).sum()
        #     self.log(f"train/loss_opacity", loss_opacity)
        #     loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        # if self.cfg.loss["lambda_scales"] > 0.0:
        #     scale_sum = torch.sum(self.geometry.get_scaling)
        #     self.log(f"train/scales", scale_sum)
        #     loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        # if self.cfg.loss["lambda_tv_loss"] > 0.0:
        #     loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
        #         out["comp_rgb"].permute(0, 3, 1, 2)
        #     )
        #     self.log(f"train/loss_tv", loss_tv)
        #     loss += loss_tv
            
        # if (
        #     out.__contains__("comp_depth")
        #     and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
        # ):
        #     loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
        #         # tv_loss(out["comp_normal"].permute(0, 3, 1, 2)) # ! 这里的comp_normal是怎么得到的
        #         tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
        #     )
        #     self.log(f"train/loss_depth_tv", loss_depth_tv)
        #     loss += loss_depth_tv

        # for name, value in self.cfg.loss.items():
        #     self.log(f"train_params/{name}", self.C(value))
            
            
        # loss_sds.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        
        iteration = self.global_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
        
        # if loss > 0:
        #     loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        # return {"loss": loss_sds}
        return {"loss": loss}
            
    
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
        #// 每次Evaluation都保存点云
        #// save_path = self.get_save_path(f"point_cloud_it{self.global_step}.ply")
        #// self.geometry.save_ply(save_path)    
    
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
        
    