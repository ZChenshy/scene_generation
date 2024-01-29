import numpy as np
import pickle
import json

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


def get_cam(bbox, 
            fovx = 60 , 
            fovy = 60 , 
            height = 512,
            width = 512,
            sample_num = 20,
            save_dir = './coarse_room/camera_config/camsInfo.pkl'):
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

    camlist = {}
    step = 0
    for campos in campositions:
        for target in target_positions:
            w2c, c2w = look_at(campos, target)
            camlist[f'{step}'] = {
                'c2w': c2w,
                'w2c': w2c,
                'fovx': fovx,
                'fovy': fovy,
                'width': width,
                'height': height,
            }
            step += 1

    with open(save_dir, 'wb')as f:
        pickle.dump(camlist,f)

    return camlist

if __name__ == '__main__':
   cams = get_cam(
       np.array([
           [-0.9685008 ,  0.05031064, -1.3602995 ],
           [ 0.81952091,  0.65031064,  1.45156923]
            ])
            )
    