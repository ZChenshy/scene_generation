import trajectory_util as tu
import numpy as np
import torch
import json

trajectory_generator = tu.left_right(0,10)

steps = 360


camera_poses = { }
for i in range(steps):
    temp=trajectory_generator(i, steps).cpu().detach().numpy()
    temp = temp.tolist()
    camera_poses[i] = temp
# 指定保存 JSON 文件的路径
print(type(camera_poses))
json_file_path = ('./left_right10_camera_trajectory.json')


# 将轨迹数据保存为 JSON 文件
with open(json_file_path, 'w') as json_file:
    json.dump(camera_poses, json_file)

print(f"Camera trajectory saved to {json_file_path}")
