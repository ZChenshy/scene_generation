import trimesh
import json
import os
import random
import numpy as np
import open3d as o3d
import plydata
from gaussiansplatting.scene.gaussian_model import BasicPointCloud


def quaternion_to_angle(quaternion):
    # To convert a quaternion matrix into angles
    x, y, z, w = quaternion
    angle = 2 * np.arctan2(np.sqrt(x**2 + y**2 + z**2), w)
    return angle


def trimesh_to_o3d(mesh):
    # 将 Trimesh 对象转换为 Open3D 中的 TriangleMesh 对象
    vertices = mesh.vertices
    faces = mesh.faces
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh


def create_box(extents, translation):
    mesh = trimesh.creation.box(extents=extents)
    mesh.vertices += translation
    return mesh


def sample_mesh_uniformly(mesh, num_points):
    # 在 mesh 表面上均匀采样点
    points = mesh.sample_points_uniformly(number_of_points=num_points)
    # 创建点云对象并设置点坐标
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points.points)

    return pointcloud


def get_vertices(bbox):
    #Given the scene's bounding box, obtain six faces represented by point cloud corresponding to this bounding box
    x1, y1, z1 = bbox[0]*1.2
    x2, y2, z2 = bbox[1]*1.2
    thickness = 0.2

    width = x2 - x1
    length = y2 - y1
    height = z2 - z1

    translation_back = [(x2 + x1) / 2, (y2 + y1) / 2, z1]
    translation_front = [(x2 + x1) / 2, (y2 + y1) / 2, z2]
    translation_top = [(x2 + x1) / 2, y2, (z2 + z1) / 2]
    translation_down = [(x2 + x1) / 2, y1, (z2 + z1) / 2]
    translation_right = [x2, (y2 + y1) / 2, (z2 + z1) / 2]
    translation_left = [x1, (y2 + y1) / 2, (z2 + z1) / 2]

    box_mesh = [
        create_box([width, length, thickness], translation_back),
        create_box([width, length, thickness], translation_front),
        create_box([width, thickness, height], translation_top),
        create_box([width, thickness, height], translation_down),
        create_box([thickness, length, height], translation_right),
        create_box([thickness, length, height], translation_left)
    ]
    pcd = [sample_mesh_uniformly(trimesh_to_o3d(mesh),3000) for mesh in box_mesh]
    return pcd


def vis_pc(pcd):
# 点云可视化
    # 获取视图控制器
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)  
    vis.add_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()
    vis.run() 
    

class scene_generator:
    def __init__(self, scene_path, models_path, room_id, scene_id, output_path = '/remote-home/share/room_gen/roomPCD'):
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
        scene_json_path = os.path.join(self.scene_path, f"{self.scene_id}.json")
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

        scene_bbox_pc = get_vertices(scene_bbox)
        ##Sequential order：back front top down right left
        extra_dict = {
            '0':'wall',
            '1':'wall',
            '2':'ceiling',
            '3':'floor',
            '4':'wall',
            '5':'wall'
            }
        
        for i, pcd in enumerate(scene_bbox_pc):
            semantic_color = self.update_color_dict(extra_dict[str(i)])
            if i==3:
                pcd = scene_bbox_pc[i]
                pcd_colors = np.repeat(np.array(semantic_color).reshape(1,-1) /255, np.asarray(pcd.points).shape[0], axis=0)
                pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
                scene.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(scene.points), np.asarray(pcd.points)]))
                scene.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(scene.colors), np.asarray(pcd.colors)]))
        

        scene_savedir = os.path.join(self.output_path, self.scene_id, self.room_id, "scene_pcd.ply")
        os.makedirs(os.path.dirname(scene_savedir), exist_ok=True)
        with open(self.color_label_dict_json , 'w') as f:
            json.dump(self.color_label_dict,f)
        o3d.io.write_point_cloud(scene_savedir, scene)
        return scene
    
    
    def update_color_dict(self, label):
        #This function is used to check whether the label has a color. If not, it will be added to the dictionary
            if label not in self.color_label_dict:
                color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                self.color_label_dict[label] = color
            else:
                color = self.color_label_dict[label]
            return color
    
    
    def load_furtinue(self, furniture,curren_room_path):
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
        theta = quaternion_to_angle(rotation_matrix)
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
    
    
    def load_scene_pcd(scene_pcd_path : str): 
        '''
            1. load the scene point cloud
            2. get the scene bounding box
            3. get the cameras extent
        '''
        if os.path.exists(os.path.join(scene_pcd_path)):
                pcd = o3d.io.read_point_cloud(scene_pcd_path)
                
                scene_info = BasicPointCloud(
                    np.array(pcd.points),
                    np.array(pcd.colors),
                    np.repeat(np.array([0,0,0]),np.array(pcd.colors).shape[0],axis=0)
                )
        else:
            print(f"[ERROR]{scene_pcd_path} is not exists")
        scene_bbox = pcd.get_axis_aligned_bounding_box()
        scene_bbox = np.vstack((scene_bbox.min_bound, scene_bbox.max_bound))
        cameras_extent = np.max((scene_bbox[1] - scene_bbox[0]) / 2)
        return scene_info, cameras_extent


def save_ply(path, save_path):
    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5
    
    data = plydata.read(path)

    xyz = np.stack((np.asarray(data.elements[0]["x"]),
                    np.asarray(data.elements[0]["y"]),
                    np.asarray(data.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(data.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(data.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(data.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:,:,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)


def load_scene_pcd(scene_pcd_path : str): 
        '''
            1. load the scene point cloud
            2. get the scene bounding box
            3. get the cameras extent
        '''
        if os.path.exists(os.path.join(scene_pcd_path)):
                pcd = o3d.io.read_point_cloud(scene_pcd_path)
                
                scene_info = BasicPointCloud(
                    np.array(pcd.points),
                    np.array(pcd.colors),
                    np.repeat(np.array([0,0,0]),np.array(pcd.colors).shape[0],axis=0)
                )
        else:
            print(f"[ERROR]{scene_pcd_path} is not exists")
        scene_bbox = pcd.get_axis_aligned_bounding_box()
        scene_bbox = np.vstack((scene_bbox.min_bound, scene_bbox.max_bound))
        cameras_extent = np.max((scene_bbox[1] - scene_bbox[0]) / 2)
        return scene_info, cameras_extent


if __name__ == '__main__':
    # scene_path=
    # models_path=
    # room_id=
    # scene_id=
    # output_path = '/remote-home/share/room_gen/roomPCD'
    # scene_pc = scene_loader()
    # print(f"Scene Point cloud has been save to : {save_path}")3
    pass