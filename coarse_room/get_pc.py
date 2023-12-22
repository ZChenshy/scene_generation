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

def run():

    def update_color_dict(label):
    #This function is used to check whether the label has a color. If not, it will be added to the dictionary
        if label not in color_label_dict:
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            color_label_dict[label] = color
        else:
            color = color_label_dict[label]
        
        return color

    ###this part will be remove in the future
    scenes_path = '/remote-home/share/room_gen/3D-FRONT-parsed'
    models_path = '/remote-home/share/room_gen/dreamGaussian_gen'
    raw_models_path = os.listdir(models_path)
    scene_id = str(raw_models_path[1])
    scene_json_path = os.path.join(scenes_path,f"{scene_id}.json")
    rooms_path = os.path.join(models_path,scene_id)
    room_path = [os.path.join(rooms_path,path) for path in os.listdir(rooms_path)]
    curren_room_path = room_path[3]
    room_instance_id = curren_room_path.split('/')[-1]
    with open(scene_json_path,'r') as f:
        scene_json = json.load(f)
    ###
    
    #load the room's json
    for i in range(len(scene_json)):
        if curren_room_path.split('/')[-1] in scene_json[i]['room_instance_id']:
            room_json = scene_json[i]
    furniture_json = room_json['room_furnitures']
    output_path = f'/remote-home/share/room_gen/room_render/{room_instance_id}'
    os.makedirs(output_path,exist_ok=True)
    color_label_dict_json = '/remote-home/abao/repositories/scene_generation/color_label_dict.json'
    if os.path.exists(color_label_dict_json):
        with open(color_label_dict_json, 'r') as file:
            color_label_dict = json.load(file)
    else:
        color_label_dict = {}

        with open(color_label_dict_json, 'w') as file:
            json.dump(color_label_dict, file, indent=4)

    scene = o3d.geometry.PointCloud()
    # load the furnitures in the scene
    for furniture in furniture_json:
        if os.path.exists(os.path.join(curren_room_path, furniture['jid'], furniture['jid'] + '.obj')):
            tr_mesh = o3d.io.read_triangle_mesh(os.path.join(curren_room_path, furniture['jid'], furniture['jid'] + '.obj'))
            points = tr_mesh.vertices
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
            raw_bbox = pcd.get_axis_aligned_bounding_box()
            bbox_vertices= np.array(furniture['bbox_vertices'])
            furniture_box = np.vstack((np.min(bbox_vertices,axis=0),np.max(bbox_vertices,axis=0)))
            scale_factors = (furniture_box[1]-furniture_box[0])/(raw_bbox.max_bound-raw_bbox.min_bound)

            points_array = np.asarray(pcd.points)
             # 对点云进行缩放
            points_array[:, 0] *= scale_factors[0]
            points_array[:, 1] *= scale_factors[1]
            points_array[:, 2] *= scale_factors[2]
            # 将缩放后的坐标重新设置回点云对象
            pcd.points = o3d.utility.Vector3dVector(points_array)

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
            semantic_color = update_color_dict(furniture['category'])
            pcd_colors = np.repeat(np.array(semantic_color).reshape(1,-1) /255, np.asarray(pcd.points).shape[0], axis=0)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
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
    semantic_color = update_color_dict(extra_dict[str(3)])
    for i,pcd in enumerate(scene_bbox_pc):
        if i==3:
            pcd = scene_bbox_pc[i]
            pcd_colors = np.repeat(np.array(semantic_color).reshape(1,-1) /255, np.asarray(pcd.points).shape[0], axis=0)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
            scene.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(scene.points), np.asarray(pcd.points)]))
            scene.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(scene.colors), np.asarray(pcd.colors)]))
    

    scene_savedir = './scene_pcd/'+curren_room_path.split("/")[-1]+'/'+'scene_pcd.ply'
    os.makedirs(os.path.dirname(scene_savedir), exist_ok=True)
    o3d.io.write_point_cloud(scene_savedir, scene)


if __name__ == '__main__':
    run()
    print('Done')