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

import os
import random
import json
import trimesh
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import BasicPointCloud
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import open3d as o3d
import numpy as np
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.fovx = 1
        self.fovy = 1
        self.width = 512
        self.height = 512
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        ###这里为修改的点云数据加载
        if os.path.exists(os.path.join(args.source_path,'scene_pcd.ply')):
            pcd = o3d.io.read_point_cloud(os.path.join(args.source_path,'scene_pcd.ply'))
            
            scene_info = BasicPointCloud(
                np.array(pcd.points),
                np.array(pcd.colors),
                np.repeat(np.array([0,0,0]),np.array(pcd.colors).shape[0],axis=0)
            )
            if os.path.exists(os.path.join(os.path.join(args.source_path,'scene_pcd.ply'))):
                pcd = trimesh.load(os.path.join(args.source_path,'scene_pcd.ply'))
                #pcd.vertices =pcd.vertices * np.array([1, 1, 1])
                pcd = normal_scene(pcd)
                pcd.show()
                ###这里为测试时使用的代码，后续需要删除
                #在点云数据中加入坐标轴，
                #pcd = add_axis(1,1,1, pcd)
                # pcd.show()
                ##############################
                scene_info = BasicPointCloud(
                    np.array(pcd.vertices),
                    np.array(pcd.colors[:, :3]),
                    np.repeat(np.array([0,0,0]),np.array(pcd.colors).shape[0],axis=0)
                )
        else:
            print(f"{args.model_path}/scene_pcd.ply is not exists")
        
        scene_bbox = pcd.get_axis_aligned_bounding_box()
        scene_bbox = np.vstack((scene_bbox.min_bound,scene_bbox.max_bound))
        self.bbox = scene_bbox
        self.cameras_extent = np.max((scene_bbox[1]-scene_bbox[0])/2)

        self.camera_position = (scene_bbox[0]+scene_bbox[1]) * np.array([0.5,0.5,0.5])
        
        self.gaussians.create_from_pcd(scene_info, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def normal_scene(scene):
        points = scene.vertices
        scene_bbox = scene.bounds
        scene_centroid = scene.centroid
        scene.vertices = scene.vertices - scene_centroid
        scale_factor = 1 / (scene_bbox[1][1] - scene_bbox[0][0])
        scene.vertices = scene.vertices  * np.array([scale_factor, scale_factor, scale_factor])
        
        return scene