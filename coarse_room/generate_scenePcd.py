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
# from coarse_room import utils
from random import randint
import math
import utils
#from utils import vis_pc

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
        
        #加载color 2 label # TODO: remove color dict
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
        scene = trimesh.Scene()

        # load the furnitures in the scene
        for furniture in self.furniture_json:
            if os.path.exists(os.path.join(furniture['path'], 'point-cloud.ply')) and ('Lamp' not in furniture['category'] if furniture['category'] is not None else True):
                print(furniture['category'])
                pcd = self.load_furtinue(furniture)
                scene.add_geometry(pcd)
            # else:
            #     raise ValueError(f"No such file or directory")
        
        scene_bbox = scene.bounds
        scene_centroid = (scene_bbox[0]+scene_bbox[1])/2
        scene_len = (scene_bbox[1]-scene_bbox[0])

        scene_bbox_pc = utils.get_vertices(scene_bbox)
        ##Sequential order：back front top down right left
        extra_dict = {
            '0':'wall0',
            '1':'wall1',
            '2':'ceiling',
            '3':'floor',
            '4':'wall2',
            '5':'wall3'
            }
        colors = [[233, 194, 123],[233, 194, 123], [192, 192, 192],  [175, 238, 238], [230, 190, 120], [230, 190, 120] ]
        #按照需求添加自己需要的
        for i, pcd in enumerate(scene_bbox_pc):
            #semantic_color = self.update_color_dict(extra_dict[str(i)])
            color = np.array(colors[i])
            pcd_colors = np.repeat(np.array(color).reshape(1,-1) /255, np.asarray(pcd.vertices).shape[0], axis=0)
            pcd.visual.vertex_colors = pcd_colors  
            scene.add_geometry(pcd)
        
        point_clouds = [trimesh.points.PointCloud(geometry.vertices, colors=geometry.visual.vertex_colors) for geometry in scene.geometry.values()]

        # 合并点云和颜色信息
        merged_vertices = np.concatenate([pcd.vertices for pcd in point_clouds])
        merged_colors = np.concatenate([pcd.visual.vertex_colors for pcd in point_clouds])

        # 创建一个新的点云，包含合并后的点和颜色信息
        merged_pcd = trimesh.points.PointCloud(merged_vertices, colors=merged_colors)

        # merged_scene = self.normal_scene(merged_pcd)
        # merged_scene.show()
        scene_savedir = os.path.join(self.output_path, self.scene_id, self.room_id, "scene_pcd_whole_wall.ply")
        os.makedirs(os.path.dirname(scene_savedir), exist_ok=True)
        with open(self.color_label_dict_json , 'w') as f:
            json.dump(self.color_label_dict,f)
        
        merged_pcd.export(scene_savedir)
        return scene
    
    def update_color_dict(self,label):
        #This function is used to check whether the label has a color. If not, it will be added to the dictionary
            if label not in self.color_label_dict:
                color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                self.color_label_dict[label] = color
            else:
                color = self.color_label_dict[label]
            return color
    
    def load_furtinue(self,furniture):
        pcd = trimesh.load(os.path.join(furniture['path'], 'point-cloud.ply'))
        pcd.vertices[:, [1, 2]] = pcd.vertices[:, [2, 1]]  #将y轴和z轴交换
        #根据模型的bbox和json里的bbox，计算缩放因子scale
        raw_bbox = pcd.bounds
        bbox_vertices= np.array(furniture['bbox_vertices'])
        furniture_box = np.vstack((np.min(bbox_vertices,axis=0),np.max(bbox_vertices,axis=0)))
        scale_factors = (furniture_box[1]-furniture_box[0])/(raw_bbox[1]-raw_bbox[0])
        
        #创建缩放矩阵
        S = np.eye(4)
        S[0, 0] = scale_factors[0]
        S[1, 1] = scale_factors[1]
        S[2, 2] = scale_factors[2]
        # 对点云进行缩放
        pcd.apply_transform(S)

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
        pcd.apply_transform(transform_matrix)

        #创建语义颜色并赋给点云
        # semantic_color = self.update_color_dict(furniture['category'])
        # pcd_colors = np.repeat(np.array(semantic_color).reshape(1,-1), np.asarray(pcd.vertices).shape[0], axis=0)
        # pcd.colors = pcd_colors

        return pcd
    
if __name__== "__main__":
    scene_path = "/remote-home/share/room_gen/3D-FRONT-parsed"
    models_path = "/remote-home/share/room_gen/shape-output"
    scene_id = "4944051f-3a7e-4387-b5f3-f925ae6da57e"
    room_id = "LivingRoom-4719"
    
    scene_loader = scene_loader(scene_path, models_path,room_id,scene_id)
    scene = scene_loader.generate_scene()
    #scene.show()
    print()