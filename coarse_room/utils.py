import numpy as np
import trimesh

def quaternion_to_angle(quaternion):
    # To convert a quaternion matrix into angles
    x, y, z, w = quaternion
    angle = 2 * np.arctan2(np.sqrt(x**2 + y**2 + z**2), w)
    return angle

def create_box(extents, translation):
    mesh = trimesh.creation.box(extents=extents)
    mesh.vertices += translation
    return mesh

def sample_mesh_uniformly(mesh, num_points):
    # 在 mesh 表面上均匀采样点
    points = mesh.sample(num_points)
    # 创建点云对象并设置点坐标
    pointcloud = trimesh.PointCloud(points)

    return pointcloud

def get_vertices(bbox):
    #Given the scene's bounding box, obtain six faces represented by point cloud corresponding to this bounding box
    scale = 1.5
    x1, y1, z1 = bbox[0] 
    x2, y2, z2 = bbox[1]

    thickness = 0.01
    width0 = (x2 - x1) * scale
    length0 =(y2 - y1) * scale
    height0 = (z2 - z1) * scale 
   
    x1, y1, z1 = bbox[0] - np.array([width0*(scale-1)/2, 0, height0*(scale-1)/2])
    x2, y2, z2 = bbox[1] + np.array([width0*(scale-1)/2, length0*(scale-1)/2, height0*(scale-1)/2])
    width = (x2 - x1) 
    length =(y2 - y1) 
    height = (z2 - z1) 
    translation_back = [(x2 + x1) / 2, (y2 + y1) / 2, z1]
    translation_front = [(x2 + x1) / 2, (y2 + y1) / 2, z2]
    translation_top = [(x2 + x1) / 2, y2, (z2 + z1) / 2]
    translation_down = [(x2 + x1) / 2, y1, (z2 + z1) / 2]
    translation_right = [x2 , (y2 + y1) / 2 , (z2 + z1) / 2]
    translation_left = [x1, (y2 + y1) / 2, (z2 + z1) / 2]

    box_mesh = [
        create_box([width, length, thickness], translation_back),
        create_box([width, length, thickness], translation_front),
        create_box([width, thickness, height], translation_top),
        create_box([width, thickness, height], translation_down),
        create_box([thickness, length, height], translation_right),
        create_box([thickness, length, height], translation_left)
    ]
    pcd = [sample_mesh_uniformly(mesh,30000) for mesh in box_mesh]
    return pcd

def add_axis(lenx, leny, lenz, origin):
        
    points_x = np.array([[x, 0, 0] for x in np.arange(0, lenx, 0.01)])  # x轴上的点云直线
    points_y = np.array([[0, y, 0] for y in np.arange(0, leny, 0.01)])  # y轴上的点云直线
    points_z = np.array([[0, 0, z] for z in np.arange(0, lenz, 0.01)])  # z轴上的点云直线

    # 将数组转换为点云
    cloud_x = trimesh.points.PointCloud(points_x)
    cloud_y = trimesh.points.PointCloud(points_y)
    cloud_z = trimesh.points.PointCloud(points_z)
    cloud_x.colors = np.tile(np.array([255, 0, 0]), (len(cloud_x.vertices), 1))  # 红色
    cloud_y.colors = np.tile(np.array([0, 255, 0]), (len(cloud_y.vertices), 1))  # 绿色
    cloud_z.colors = np.tile(np.array([0, 0, 255]), (len(cloud_z.vertices), 1))  # 蓝色
    axis = [cloud_x, cloud_y, cloud_z]

    # 合并两个点云
    merged_points = np.concatenate([origin.vertices] + [cloud.vertices for cloud in axis])
    merged_colors = np.concatenate([origin.colors] + [cloud.colors for cloud in axis])


    # 创建一个新的PointCloud对象
    merged_origin = trimesh.points.PointCloud(merged_points, merged_colors)
    return merged_origin


if __name__ == '__main__':
    bbox = np.array([
        [-1,-1,-1],
        [2,3,4]
                    ])
    pcd = get_vertices(bbox)