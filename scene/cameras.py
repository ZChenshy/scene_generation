#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from random import randint
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getWorld2View,getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class customCamera(nn.Module):
    def __init__(self, c2w, FoVx, FoVy, 
                 width = 512, height = 512,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", 
                 ):
        super(customCamera, self).__init__()

        
        self.c2w = c2w
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        w2c = np.linalg.inv(c2w)
        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class camera_generator:
    def __init__(self,scene_bbox,num_cameras,fovx,fovy,width,height):
        self.fovx = fovx
        self.fovy = fovy
        self.width = width
        self.height = height
        self.bbox = scene_bbox
        self.camera_position = (scene_bbox[0]+scene_bbox[1]) * np.array([0.5,0.5,0.5])
        self.num_cameras = num_cameras
        self.scene_vertices = self.get_vertices()
        self.bottom_vertices = np.array([value for key,value in self.scene_vertices.items() if '上' in key])
    
    def genCameras(self,opengl = True):
        viewpoint_stack = []
        for target_point in self.target_point():
            c2w_martix = self.camera_pose(target_point,opengl)
            viewpoint_cam = customCamera(c2w_martix,self.fovx,self.fovy,self.width,self.height)
            viewpoint_stack.append(viewpoint_cam)
        return viewpoint_stack


    def target_point(self):
        step = 360/self.num_cameras
        angel = [randint(i*step,(i+1)*step) for i in range(self.num_cameras)]
        target_point = [self.select_point_on_edge(self.bottom_vertices,angel[i]) for i in range(self.num_cameras)]
        return target_point
        #target_point *= np.array([0.8,0.8,0.8])
    
    def select_point_on_edge(self,rectangle_points, angle):
        """
        在矩形的边上根据给定角度选择一个点。
        
        :param rectangle_points: 矩形的四个顶点坐标，格式为 [(x1, y, z1), (x2, y, z2), (x3, y, z3), (x4, y, z4)]
        :param angle: 角度，范围从0到360度
        :return: 在矩形边上的点的坐标
        """
        # 确定矩形的边
        edges = [(rectangle_points[i], rectangle_points[(i + 1) % len(rectangle_points)]) for i in range(len(rectangle_points))]

        # 将角度转换为弧度
        angle_rad = np.deg2rad(angle)

        # 根据角度选择边
        edge_index = int(angle // 90) % 4
        edge = edges[edge_index]

        # 在选定的边上找到点
        dx, dy, dz = np.array(edge[1]) - np.array(edge[0])
        t = (angle % 90) / 90  # 边上的插值参数
        point = np.array(edge[0]) + np.array([t * dx, t * dy, t * dz])

        return point

    def get_vertices(self):

        # corner1 和 corner2 是包围盒对角线上的两个点
        x1, y1, z1 = self.bbox[0]
        x2, y2, z2 = self.bbox[1]
        vertices = {
            "左上后": np.array([x1, y2, z1]),
            "左上前": np.array([x1, y2, z2]),
            "左下后": np.array([x1, y1, z1]),
            "左下前": np.array([x1, y1, z2]),
            "右下前": np.array([x2, y1, z2]),
            "右上前": np.array([x2, y2, z2]),
            "右下后": np.array([x2, y1, z1]),
            "右上后": np.array([x2, y2, z1])
        }
        return vertices
    
    def camera_pose(self,target_position,opengl=True):
        T = np.eye(4, dtype=np.float32)
        campos =self.camera_position
        T[:3, :3] = self.look_at(campos,target_position, opengl)
        T[:3, 3] = self.camera_position
        return T

    def look_at(self,campos,target, opengl=False):
        # campos: [N, 3], camera/eye position
        # target: [N, 3], object to look at
        # return: [N, 3, 3], rotation matrix
        if not opengl:
            # camera forward aligns with -z
            forward_vector = safe_normalize(target - campos)
            up_vector = np.array([0, -1, 0], dtype=np.float32)
            right_vector = safe_normalize(np.cross(forward_vector, up_vector))
            up_vector = safe_normalize(np.cross(right_vector, forward_vector))
        else:
            # camera forward aligns with +z
            forward_vector = safe_normalize(campos - target)
            up_vector = np.array([0, -1, 0], dtype=np.float32)
            right_vector = safe_normalize(np.cross(up_vector, forward_vector))
            up_vector = safe_normalize(np.cross(forward_vector, right_vector))
        R = np.stack([right_vector, up_vector, forward_vector], axis=1)
        return R

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)
# elevation & azimuth to pose (cam2world) matrix
# def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
#     # radius: scalar
#     # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
#     # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
#     # return: [4, 4], camera pose matrix
#     if is_degree:
#         elevation = np.deg2rad(elevation)
#         azimuth = np.deg2rad(azimuth)
#     x = radius * np.cos(elevation) * np.sin(azimuth)
#     y = - radius * np.sin(elevation)
#     z = radius * np.cos(elevation) * np.cos(azimuth)
#     if target is None:
#         target = np.zeros([3], dtype=np.float32)
#     campos = np.array([x, y, z]) + target  # [3]
#     T = np.eye(4, dtype=np.float32)
#     T[:3, :3] = look_at(campos, target, opengl)
#     T[:3, 3] = campos
#     return T




