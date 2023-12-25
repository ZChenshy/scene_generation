import trimesh
import json
import pickle
import os
import cv2
import random
from PIL import Image
import numpy as np
from trimesh.transformations import rotation_matrix, transform_points
#from trajectory import get_extrinsics,_rot_xyz
import open3d as o3d
import utils
from random import randint
import math
from utils import vis_pc

class scene_loader:
    def __init__(self,scene_path,models_path,room_id,scene_id, output_path = '/remote-home/share/room_gen/roomPCD'):
        #保存场景的json信息
        self.scene_path = scene_path

        #保存家具的model文件
        self.models_path = models_path

        self.room_id = room_id
        self.scene_id = scene_id
        self.output_path = output_path
        #当前房间的家具文件的路径
        self.room_models_path = os.path.join(self.models_path,self.scene_id,self.room_id)
        self.parser_scene()
  
    def parser_scene(self):
        #加载房间的json信息
        scene_json_path = os.path.join(self.scene_path,f"{self.scene_id}.json")
        with open(scene_json_path,'r') as f:
            scene_json = json.load(f)

        for room_json in scene_json:
            if self.room_id in room_json['room_instance_id']:
                self.room_json = room_json
                self.furniture_json = room_json['room_furnitures']
        
        #加载color 2 label
        self.color_label_dict_json = os.path.join(self.output_path,'color_label_dict.json')
        os.makedirs(self.output_path, exist_ok=True)
        if not os.path.exists(self.color_label_dict_json):
            color_label_dict = {}
            with open(self.color_label_dict_json, 'w') as file:
                json.dump(color_label_dict, file, indent=4)
        else:
            with open(self.color_label_dict_json, 'r') as file:
                color_label_dict = json.load(file)
        self.color_label_dict = color_label_dict

    def generate_scene(self):
        scene = o3d.geometry.PointCloud()

        # load the furnitures in the scene
        curren_room_path = os.path.join(self.models_path,self.scene_id,self.room_id)
        for furniture in self.furniture_json:
            if os.path.exists(os.path.join(curren_room_path, furniture['jid'], furniture['jid'] + '.obj')):
                pcd = self.load_furtinue(furniture,curren_room_path)
                scene.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(scene.points), np.asarray(pcd.points)]))
                scene.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(scene.colors), np.asarray(pcd.colors)]))
            else:
                raise ValueError(f"No such file or directory: {os.path.join(curren_room_path, furniture['jid'], furniture['jid'] + '.obj')}")
        
        scene_bounding_box = scene.get_axis_aligned_bounding_box()
        scene_bbox = np.vstack((scene_bounding_box.min_bound,scene_bounding_box.max_bound))
        scene_centroid = (scene_bbox[0]+scene_bbox[1])/2
        scene_len = (scene_bbox[1]-scene_bbox[0])

        scene_bbox_pc = utils.get_vertices(scene_bbox)
        ##Sequential order：back front top down right left
        extra_dict = {
            '0':'wall',
            '1':'wall',
            '2':'ceiling',
            '3':'floor',
            '4':'wall',
            '5':'wall'
            }
        #按照需求添加自己需要的
        for i,pcd in enumerate(scene_bbox_pc):
            semantic_color = self.update_color_dict(extra_dict[str(i)])
            if i==3:
                pcd = scene_bbox_pc[i]
                pcd_colors = np.repeat(np.array(semantic_color).reshape(1,-1) /255, np.asarray(pcd.points).shape[0], axis=0)
                pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
                scene.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(scene.points), np.asarray(pcd.points)]))
                scene.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(scene.colors), np.asarray(pcd.colors)]))
    
        
        
        scene = self.normal_scene(scene)
        scene_savedir = os.path.join(self.output_path,self.scene_id,self.room_id,"scene_pcd.ply")
        os.makedirs(os.path.dirname(scene_savedir), exist_ok=True)
        with open(self.color_label_dict_json , 'w') as f:
            json.dump(self.color_label_dict,f)
        o3d.io.write_point_cloud(scene_savedir, scene)
        return scene
    
    def normal_scene(self,scene):
        points = scene.points
        scene_bounding_box = scene.get_axis_aligned_bounding_box()
        scene_bbox = np.vstack((scene_bounding_box.min_bound,scene_bounding_box.max_bound))
        scale_factor = 1.0 / (scene_bbox[1][1] - scene_bbox[0][1])  #y轴是高度轴,将y轴缩放到长度为1，其他轴按比例缩放
        scaled_points = (points - scene_bbox[0]) * np.array([scale_factor, -scale_factor, scale_factor])
        
        #平移点云
        new_min = scaled_points.min(axis=0)
        new_max = scaled_points.max(axis=0)
        bottom_center = np.array([new_min[0] + new_max[0], 2 * new_min[1], new_min[2] + new_max[2]]) / 2
        translated_points = scaled_points - bottom_center
        scene.points = o3d.utility.Vector3dVector(translated_points)

        return scene

    def update_color_dict(self,label):
        #This function is used to check whether the label has a color. If not, it will be added to the dictionary
            if label not in self.color_label_dict:
                color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                self.color_label_dict[label] = color
            else:
                color = self.color_label_dict[label]
            return color
    
    def load_furtinue(self,furniture,curren_room_path):
        tr_mesh = o3d.io.read_triangle_mesh(os.path.join(curren_room_path, furniture['jid'], furniture['jid'] + '.obj'))
        points = tr_mesh.vertices
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=0.05)

        #根据模型的bbox和json里的bbox，计算缩放因子scale
        raw_bbox = pcd.get_axis_aligned_bounding_box()
        bbox_vertices= np.array(furniture['bbox_vertices'])
        furniture_box = np.vstack((np.min(bbox_vertices,axis=0),np.max(bbox_vertices,axis=0)))
        scale_factors = (furniture_box[1]-furniture_box[0])/(raw_bbox.max_bound-raw_bbox.min_bound)
        
        # 对点云进行缩放
        points_array = np.asarray(pcd.points)
        points_array[:, 0] *= scale_factors[0]
        points_array[:, 1] *= scale_factors[1]
        points_array[:, 2] *= scale_factors[2]

        # 将缩放后的坐标重新设置回点云对象
        pcd.points = o3d.utility.Vector3dVector(points_array)

        #旋转家具
        translation = furniture['pos']
        rotation_matrix = furniture['rot']
        theta = utils.quaternion_to_angle(rotation_matrix)
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = translation
        pcd.transform(transform_matrix)

        #创建语义颜色并赋给点云
        semantic_color = self.update_color_dict(furniture['category'])
        pcd_colors = np.repeat(np.array(semantic_color).reshape(1,-1) /255, np.asarray(pcd.points).shape[0], axis=0)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

        return pcd
    
if __name__== "__main__":
    scene_path = "/remote-home/share/room_gen/3D-FRONT-parsed"
    models_path = "/remote-home/share/room_gen/dreamGaussian_gen"
    scene_id = "4944051f-3a7e-4387-b5f3-f925ae6da57e"
    room_id = "SecondBedroom-1338"
    scene =scene_loader(scene_path,models_path,room_id,scene_id)
    _ = scene.generate_scene()
    


