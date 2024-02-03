import bisect
import math
from dataclasses import dataclass, field
import pickle
from random import choice
import numpy as np
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

def sample_points_on_rectangle(bbox, num):
    corner1, corner2 = bbox
    # 计算矩形的四个顶点
    p1 = np.array(corner1)
    p2 = np.array([corner2[0], corner1[1], corner1[2]])
    p3 = np.array([corner2[0], corner1[1], corner2[2]])
    p4 = np.array([corner1[0], corner1[1], corner2[2]])

    # 计算每条边的长度
    length1 = np.linalg.norm(p2 - p1)
    length2 = np.linalg.norm(p3 - p2)

    # 总周长
    perimeter = 2 * (length1 + length2)
    # 根据周长比例分配点数
    num_side1 = int(num * length1 / perimeter)
    num_side2 = int(num * length2 / perimeter)

    # 生成边上的点
    points = []
    for i in range(num_side1 + 1):
        points.append(p1 + i * (p2 - p1) / num_side1)
    for i in range(num_side2 + 1):
        points.append(p2 + i * (p3 - p2) / num_side2)
    for i in range(num_side1 + 1):
        points.append(p3 + i * (p4 - p3) / num_side1)
    for i in range(num_side2 + 1):
        points.append(p4 + i * (p1 - p4) / num_side2)

    return points


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
    
    w2c = np.identity(4)
    c2w = np.identity(4)

    w2c[:3, :3] = R
    w2c[:3, 3] = R @ (-campos)

    c2w[:3, :3] = R.T
    c2w[:3, 3] = campos
    
    return w2c, c2w


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
    eval_fovy_deg: float = 40
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy
    ###
    #该部分参数用于test部分，固定相机在某个位置，旋转相机观察场景
    rota_camera: bool = False #该参数用于控制使用旋转相机
    camera_position: Tuple[float, float, float] = (0, 0.1, 0) #相机固定位置
    camera_eval: float = -45 #相机固定仰角，负值相机向下看，正值相机向上看
    rotation_angle: float = 360 #相机旋转角度，360度为一圈

    ###
    #该部分参数用于test部分，相机轨迹为一个圆，相机在圆上运动，观察场景
    round_camera: bool = True 
    round_center: Tuple[float, float, float] = (0.03, 0.03, 0.1) #相机轨迹的圆心
    radius: float = 0.03 #相机轨迹的半径
    look_direction: str = 'outside' #相机看向的方向，inside看向圆心，outside看向圆外
    
    camera_trajectory: str = 'move'
    target_trajectory: str = 'move'

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
        self.radius = self.cfg.radius
        self.look_direction = self.cfg.look_direction

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
                camera_positions = torch.tensor([self.camera_position]).repeat(self.n_views, 1)
                
                elevation_deg = self.camera_eval 
                rotation_angle = torch.linspace(0., self.rotation_angle, self.n_views)
                radius = math.fabs(torch.tensor(self.camera_position)[1]) / math.tan(math.fabs(elevation_deg) * math.pi / 180.0)
                center = torch.tensor([self.camera_position[0], 0, self.camera_position[2]]).repeat(self.n_views, 1)
                target_display = torch.stack([torch.cos(rotation_angle * math.pi / 180.0) * radius, torch.zeros_like(rotation_angle), torch.sin(rotation_angle * math.pi / 180.0) * radius], dim=1)
                target_points = target_display + center

            ###################
            ###################
            elif self.round_camera:
                round_center = torch.tensor([self.round_center]).repeat(self.n_views, 1)
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
        up: Float[Tensor, "B 3"] = torch.as_tensor([0 , -1 , 0], dtype=torch.float32)[None, :].repeat(self.n_views, 1)
        lookat: Float[Tensor, "B 3"] = F.normalize(target_points - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)

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

        fovy_deg: Float[Tensor, "B"] = torch.tensor(self.cfg.eval_fovy_deg).repeat(self.n_views)
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

        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.w2c = w2c
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = 45, 45
        self.elevation_deg, self.azimuth_deg = 45,45
        self.camera_distances = 0.3
        self.fovy = fovy


    def __len__(self):
        return self.n_views


    def __getitem__(self, index):
        return {
            "index": index,
            "mvp_mtx": self.mvp_mtx[index],
            "w2c": self.w2c[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distances,
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
            "fovy":self.fovy[index],
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch



@register("outward-camera-datamodule") 
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDatasetSingle(self.cfg)
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
    
class RandomCameraIterableDatasetSingle(IterableDataset, Updateable):
    """
    具体实现在collate
    """
    def _get_panorama_cameras(
        self,
        camerapositions, 
        fovx = 60, 
        fovy = 60 , 
        height = 1024,
        width = 1024,
        save_dir = 'coarse_room/camera_config/panocamsInfo.pkl'
    ):
        """
        用于生成拼接相机。必须要传入的参数是camerapositions。
        环绕场景旋转, 每次旋转的角度为fovx。
        """
        radius = 0.5
        rot_angle = fovx * np.pi / 180
        theta = np.linspace(0, 2 * np.pi, int(2 * np.pi / rot_angle ), endpoint=False)
        angle_random = np.full_like(theta, np.random.rand(1) * np.pi /2) 
        theta = (theta + angle_random)
        camdict = {}
        c2w_list = []
        w2c_list = []
        fovx_list = []
        fovy_list = []
        for rot in theta:
            dis = np.array([radius * np.cos(rot), 0, radius * np.sin(rot)])
            target_positions = camerapositions + dis
            w2c, c2w = look_at(camerapositions, target_positions)
            c2w_list.append(c2w)
            w2c_list.append(w2c)
            fovx_list.append(fovx)
            fovy_list.append(fovy)     
                
        camdict = {
            "c2w": c2w_list,
            "w2c": w2c_list,
            "fovx": fovx_list,
            "fovy": fovy_list,
            "width": width,
            "height": height,
        }

        with open(save_dir, 'wb')as f:
            pickle.dump(camdict,f)

        return camdict
    
    def _get_cam( 
            self,
            bbox,
            fovx = 60 , 
            fovy = 60 , 
            height = 1024,
            width = 1024,
            sample_num = 20,
            save_dir = 'coarse_room/camera_config/camsInfo.pkl'
        ):
        """
        传入参数：场景的bbox！！！（这个是必须的）
        其他：fovx，fovy(角度制);image_height，image_width。未传入将使用默认的。
        sample_num 决定采样多少个相机位置
        .........................
        .  +++++++++++++++++++  .
        .  +                 +  .
        .  +      ******     +  .
        .  +      *    *     +  .
        .  +      ******     +  .
        .  +++++++++++++++++++  .
        .........................
        最外围由'.'组成的矩形是房间地面的bbox。由'+'组成的矩形是相机运动的轨迹。
        由'*'构成的矩形的四个顶点是相机看向的目标点。
        首先根据地面的bbox，适当缩放，得到相机运动轨迹的矩形，然后在轨迹上均匀采样，
        获得相机的位置campositions。然后再次对地面的bbox进行缩放，得到一个更小的矩形，
        将这个矩形的四个顶点作为相机观察的目标点。此外，我们还额外将对角线交点也作为相机观察
        的目标点。
        """
        
        fovx = fovx * np.pi /180
        fovy = fovy * np.pi /180

        len = bbox[1][1]-bbox[0][1]
        #相机轨迹对应的矩形
        trajectory_box = bbox * np.array([0.95, 1, 0.95]) + np.array([[0, 0.3*len, 0],[0, -0.1*len, 0]])
        
        #相机观察点对应的矩形
        corner1, corner2 = bbox * np.array([0.5, 1, 0.5]) + np.array([[0, 0.3*len, 0],[0, -0.1*len, 0]])
        # 计算四个观察点
        p1 = np.array(corner1)
        p2 = np.array([corner2[0], corner1[1], corner1[2]])
        p3 = np.array([corner2[0], corner1[1], corner2[2]])
        p4 = np.array([corner1[0], corner1[1], corner2[2]])
        target_positions = np.array([p1, p2, p3, p4])
        campositions = sample_points_on_rectangle(trajectory_box, sample_num)

        camdict = {}
        c2w_list = []
        w2c_list = []
        fovx_list = []
        fovy_list = []

        for campos in campositions:
            for target in target_positions:
                w2c, c2w = look_at(campos, target)
                c2w_list.append(c2w)
                w2c_list.append(w2c)
                fovx_list.append(fovx)
                fovy_list.append(fovy)
                
        camdict = {
            "c2w": c2w_list,
            "w2c": w2c_list,
            "fovx": fovx_list,
            "fovy": fovy_list,
            "width": width,
            "height": height,
        }

        with open(save_dir, 'wb')as f:
            pickle.dump(camdict,f)

        return camdict

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        
        self.camDict = self._get_cam(bbox=np.array([
                [-0.9685008 ,  0.05031064, -1.3602995 ],
                [ 0.81952091,  0.65031064,  1.45156923],
            ]))
        # TODO：在这里写入相机的位置信息，房间的BBOX
        self.panocamDict = self._get_panorama_cameras(camerapositions=np.array([0, 0, 0]))
        self.init = True # 用于控制是否是第一次迭代，第一次迭代时，将所有位置的深度图进行渲染
        self.single_iter = 100 # 每个角度对应的相机参数的返回次数
        self.cam_iter_count = {i: 0 for i in range(len(self.camDict["fovy"]))} 

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        if self.init:
            if len(self.cam_iter_count) != 0:
                viewpoint_cam_idx = choice(list(self.cam_iter_count.keys()))
                self.cam_iter_count[viewpoint_cam_idx] += 1
                if self.cam_iter_count[viewpoint_cam_idx] == 1:
                    del self.cam_iter_count[viewpoint_cam_idx]
            else:
                self.cam_iter_count = {i: 0 for i in range(len(self.camDict["fovy"]))}
                self.init = False
                
                viewpoint_cam_idx = choice(list(self.cam_iter_count.keys()))
                self.cam_iter_count[viewpoint_cam_idx] += 1
                if self.cam_iter_count[viewpoint_cam_idx] == self.single_iter:
                    del self.cam_iter_count[viewpoint_cam_idx]
        else:     
            if len(self.cam_iter_count) != 0:
                viewpoint_cam_idx = choice(list(self.cam_iter_count.keys()))
                self.cam_iter_count[viewpoint_cam_idx] += 1
            if self.cam_iter_count[viewpoint_cam_idx] == self.single_iter:
                del self.cam_iter_count[viewpoint_cam_idx]
                
        return {
            "c2w": torch.tensor(self.camDict["c2w"][viewpoint_cam_idx]).unsqueeze(0).to(device=get_device()),
            "w2c": torch.tensor(self.camDict["w2c"][viewpoint_cam_idx]).unsqueeze(0).to(device=get_device()),
            "fovx": [self.camDict["fovx"][viewpoint_cam_idx]],
            "fovy": [self.camDict["fovy"][viewpoint_cam_idx]],
            "width": self.camDict["width"],
            "height": self.camDict["height"],
            "cam_idx": viewpoint_cam_idx
        }
        
    