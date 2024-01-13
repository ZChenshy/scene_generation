import bisect
import math
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

import os
import numpy as np

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

@dataclass
class RandomCameraDataModuleConfig:
    
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 512
    width: Any = 512
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    fix: bool = True
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 60)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (4.,6.)
    fix_elevation_deg: float = 90
    fix_azimuth_deg: float = 0
    fix_fovy_deg: float = 60
    fix_camera_distance: float = 4.0
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.
    center_perturb: float = 0.
    up_perturb: float = 0.0
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 6.
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy
    ###
    #该部分参数用于test部分，固定相机在某个位置，旋转相机观察场景
    rota_camera: bool = True #该参数用于控制使用旋转相机
    camera_position: Tuple[float, float, float] = (0, 0.25, 0) #相机固定位置
    camera_eval: float = -45 #相机固定仰角，负值相机向下看，正值相机向上看
    rotation_angle: float = 360 #相机旋转角度，360度为一圈

    ###
    #该部分参数用于test部分，相机轨迹为一个圆，相机在圆上运动，观察场景
    round_camera: bool = False 
    round_center: Tuple[float, float, float] = (0, 0.25, 0) #相机轨迹的圆心
    radius: float = 0.1 #相机轨迹的半径
    look_direction: str = 'outside' #相机看向的方向，inside看向圆心，outside看向圆外
    


class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        self.rota_camera = self.cfg.rota_camera
        self.camera_position = self.cfg.camera_position
        self.camera_eval = self.cfg.camera_eval
        self.rotation_angle = self.cfg.rotation_angle

        self.round_camera = self.cfg.round_camera
        self.round_center = self.cfg.round_center
        self.radius = self.radius
        self.look_direction = self.look_direction

        

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0., 360.0, self.n_views + 1)[: self.n_views]
            elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
            )
            camera_distances: Float[Tensor, "B"] = torch.full_like(
                elevation_deg, self.cfg.eval_camera_distance
            )

            elevation = elevation_deg * math.pi / 180
            azimuth = azimuth_deg * math.pi / 180

            # convert spherical coordinates to cartesian coordinates
            # right hand coordinate system, z front, x right, y up
            # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.sin(elevation),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    
                ],
                dim=-1,
            )

            # default scene center at origin
            target_points: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        else:
            ###################
            if self.rota_camera:
                camera_positions = torch.tensor([self.camera_position]).repeat(self.n_views)

                elevation_deg = self.camera_eval 
                rotation_angle = torch.linspace(0., self.rotation_angle, self.n_views)
                radius = math.fabs(self.camera_position)[1] / math.tan(math.fabs(elevation_deg) * math.pi / 180.0)
                center = torch.tensor([self.camera_position[0], 0, self.camera_position[2]]).repeat(self.n_views)
                target_display = torch.stack([torch.cos(rotation_angle * math.pi / 180.0) * radius, torch.zeros_like(rotation_angle), torch.sin(rotation_angle * math.pi / 180.0) * radius], dim=1)
                target_points = target_display + center

            ###################
            ###################
            elif self.round_camera:
                round_center = torch.tensor([self.round_center]).repeat(self.n_views)
                radius = self.radius
                rotation_angle = torch.linspace(0., 360., self.n_views)
                display = torch.stack([torch.cos(rotation_angle * math.pi / 180.0) * radius, torch.zeros_like(rotation_angle), torch.sin(rotation_angle * math.pi / 180.0) * radius], dim=1)
                camera_positions = display + round_center
                if self.look_direction == "outside":
                    radius_target = radius *1.5
                    display_targets = torch.stack([torch.cos(rotation_angle * math.pi / 180.0) * radius_target, torch.zeros_like(rotation_angle), torch.sin(rotation_angle * math.pi / 180.0) * radius_target], dim=1)
                    target_points = display_targets + round_center
                elif self.look_direction == "inside":
                    target_points = round_center
                else:
                    raise ValueError("look_direction only supports 'outside' and 'inside'")
            ###################
        
        # default camera up direction as +y
        lookat: Float[Tensor, "B 3"] = F.normalize(target_points - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up,dim=-1), dim=-1)

        #修正up向量
        up = F.normalize(torch.cross(right, lookat,dim=-1), dim=-1)

        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        #这里也可以用c2w的逆来获取w2c，但是这种方式计算出的w2c会有一点偏差
        w2c3x4 = torch.cat(
            [torch.stack([right, up, lookat], dim=1), torch.stack([right, up, lookat], dim=1) @ (-camera_positions[:, :, None])],
            dim=-1,
        )
        w2c = torch.cat(
            [w2c3x4, torch.zeros_like(w2c3x4[:, :1])], dim=1
        )
        w2c[:, 3, 3] = 1.0

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        c2w_3dgs = []
        for id in range(self.n_views):
            # TODO : render_pose fix
            # origin code : render_pose = pose_spherical( azimuth_deg[id] + 180.0 - self.load_type*90, -elevation_deg[id], camera_distances[id])
            render_pose = pose_spherical( azimuth_deg[id] + 180.0, -elevation_deg[id], camera_distances[id])
            
            matrix = torch.linalg.inv(render_pose)
            # R = -np.transpose(matrix[:3,:3])
            # R = -np.transpose(matrix[:3,:3])
            R = -torch.transpose(matrix[:3,:3], 0, 1)
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]
            c2w_single = torch.cat([R, T[:,None]], 1)
            c2w_single = torch.cat([c2w_single, torch.tensor([[0,0,0,1]])], 0)
            # c2w_single = convert_camera_to_world_transform(c2w_single)
            c2w_3dgs.append(c2w_single)
        c2w_3dgs = torch.stack(c2w_3dgs, 0)
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.w2c = w2c
        self.c2w_3dgs = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.fovy = fovy

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "mvp_mtx": self.mvp_mtx[index],
            "w2c": self.w2c[index],
            "c2w": self.c2w[index],
            "c2w_3dgs": self.c2w_3dgs[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
            "fovy":self.fovy[index],
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch



@register("random-outward-camera-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDatasetCustom(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
    
class RandomCameraIterableDatasetCustom(IterableDataset, Updateable):
    """
    具体实现在collate
    """
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]

        self.fix = True #该参数为true，则固定相机位置，否则不固定。
        
        #使用固定的相机位置、相机焦距
        self.fix_elevation_deg = self.cfg.fix_elevation_deg
        self.fix_azimuth_deg = self.cfg.fix_azimuth_deg
        self.fix_camera_distance = self.cfg.fix_camera_distance
        self.fix_fovy_deg = self.cfg.fix_fovy_deg
       
        #使用不固定相机位置
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range
        

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        pass

        # r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        # self.elevation_range = [
        #     (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
        #     (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        # ]
        # self.azimuth_range = [
        #     (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
        #     (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        # ]

        # self.camera_distance_range = [
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[0],
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[1],
        # ]
        # self.fovy_range = [
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[0],
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[1],
        # ]

    def collate(self, batch) -> Dict[str, Any]:
        """
        使用球坐标系表示相机位置, 由elevation.azimuth,camera_distance三个参数确定相机的位置。
        默认以原点为球心, camera_distance为半径。固定elevation为 pi/2(90度)，即可固定相机的位置
        在本场景表示中, z轴是向上的, 这点与原始3dgs是不同的

        """
        fovy_deg: Float[Tensor, "B"]
        fovy: Float[Tensor, "B"]
        fovy_deg = torch.full((self.batch_size,), self.fix_fovy_deg)
        fovy = fovy_deg * math.pi / 180


        if self.fix ==True:
            """
            使用固定的相机位置
            """
            elevation_deg = torch.tensor(self.fix_elevation_deg).repeat(self.batch_size)
            elevation = elevation_deg * math.pi / 180
            azimuth_deg = torch.randint(low=0, high=360, size = (self.batch_size,))
            azimuth = azimuth_deg * math.pi / 180
            camera_distances = torch.tensor(self.fix_camera_distance).repeat(self.batch_size)
            camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth), 
                camera_distances * torch.sin(elevation),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),     
            ], dim=-1)
            center = torch.zeros_like(camera_positions)
            
        else:
            elevation_deg: Float[Tensor, "B"]
            elevation: Float[Tensor, "B"]
            elevation_deg = (
                torch.rand(self.batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        
            
            azimuth_deg: Float[Tensor, "B"]
            azimuth: Float[Tensor, "B"]
            azimuth_deg = (
                torch.rand(self.batch_size) + torch.arange(self.batch_size)
            ) / self.batch_size * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[0]
            azimuth = azimuth_deg * math.pi / 180

            camera_distances: Float[Tensor, "B"]
            camera_distances: Float[Tensor, "B"] = (
                torch.rand(self.batch_size)
                * (self.camera_distance_range[1] - self.camera_distance_range[0])
                + self.camera_distance_range[0]
            )


            #注意坐标系的不同。在这里，y 对应 up，x 对应 right，z 对应 front
            camera_positions = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth), 
                    camera_distances * torch.sin(elevation),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth), 
                    
                ],
                dim=-1,
            )

            # default scene center at origin
            # 默认初始场景中心放置在原点
            # 场景的长宽高，把高标准化到[0,1]
            center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
            # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
            camera_perturb: Float[Tensor, "B 3"] = (
                torch.rand(self.batch_size, 3) * 2 * self.cfg.camera_perturb
                - self.cfg.camera_perturb
            )
            camera_positions = camera_positions + camera_perturb


        #生成相机的观察目标
            #target = center + [dx_dis * x_sign, dy_dis, dz_dis* z_sign]
            #dx_dis 与 dy_dis使用正态分布采样，取值范围在[0, 2]
            #dz_dis 采取均匀分布采样，取值范围在[0, 0.4]
            #x_sign, z_sign 根据 旋转角度来确定象限符号
            #根据需求手动设定dx_dis, dy_dis, dz_dis的大小，以此来调节相机仰角。
            #具体来说，dz_dis越大，相机仰角就越大。
        target_points: Float[Tensor, "B 3"] = center.clone()

        for i in range(self.batch_size):
            angle = torch.rand(1) * (2 * math.pi)
            dx_dis = torch.normal(0.15, 0.1, size = ())
            dy_dis = torch.normal(0.15, 0.1, size = ()) 
            dx_dis = torch.clamp(dx_dis, 0, 0.15) * math.cos(angle)
            dy_dis = torch.clamp(dy_dis, 0, 0.15) * math.sin(angle)
            dz_dis = torch.rand(1) * 0.15
            target_points[i, :] = target_points[i, :] + torch.tensor([dx_dis, dy_dis, dz_dis])
        
        # default camera up direction as +z
        # 如果实际渲染结果，画面是颠倒的，可以把up向量[0,0,1]调节为[0,0,-1]
        up: Float[Tensor, "B 3"] = torch.as_tensor([0 , -1 , 0], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        lookat: Float[Tensor, "B 3"] = F.normalize(target_points - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up,dim=-1), dim=-1)

        #修正up向量
        up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)

        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        #这里也可以用c2w的逆来获取w2c，但是这种方式计算出的w2c会有一点偏差
        w2c3x4 = torch.cat(
            [torch.stack([right, up, lookat], dim=1), torch.stack([right, up, lookat], dim=1) @ (-camera_positions[:, :, None])],
            dim=-1,
        )
        w2c = torch.cat(
            [w2c3x4, torch.zeros_like(w2c3x4[:, :1])], dim=1
        )
        w2c[:, 3, 3] = 1.0


        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        )

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(self.batch_size, 3) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.batch_size) * math.pi * 2 - math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        self.proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, self.proj_mtx)
        self.fovy = fovy

        return {
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "w2c": w2c,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": self.fovy,
            "proj_mtx": self.proj_mtx,
        }
    
    
def look_at(campos,target):
    """
    w2c:三个基向量按行排列
    c2w:三个基向量按列排列

    """
    norm = np.linalg.norm(target - campos)
    forward_vector = (target - campos) / norm if norm != 0 else (target - campos)
    up_vector = np.array([0, -1, 0], dtype=np.float32)
    norm = np.linalg.norm(np.cross(forward_vector, up_vector))
    right_vector = np.cross(forward_vector, up_vector) / norm if norm != 0 else (np.cross(forward_vector, up_vector))
    norm = np.linalg.norm(np.cross(right_vector, forward_vector))                                          
    up_vector = np.cross(right_vector, forward_vector) / norm if norm != 0 else np.cross(right_vector, forward_vector)
    R = np.stack([right_vector, up_vector, forward_vector], axis=0)
    print(forward_vector)
    print(forward_vector)
    print(R)
    
    w2c = np.identity(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = R @ (-campos)
    return w2c
    