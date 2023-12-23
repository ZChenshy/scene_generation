import numpy as np
import open3d as o3d
import trimesh

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

if __name__ == '__main__':
    bbox = np.array([
        [-1,-1,-1],
        [2,3,4]
                    ])
    pcd = get_vertices(bbox)

    for i in range(6):
        vis_pc(pcd[i])