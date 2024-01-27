# pip install transformers accelerate -q
# pip install git+https://github.com/huggingface/diffusers@@shap-ee


import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif
import trimesh
import numpy as np
from flask import Flask, request, jsonify
import os
from datetime import datetime
import json

app = Flask(__name__)


def MeshDecoder_2_mesh(MeshDecoder_Output):
    vertices = MeshDecoder_Output.verts.cpu().detach().numpy()
    faces = MeshDecoder_Output.faces.cpu().detach().numpy()
    R = MeshDecoder_Output.vertex_channels['R'].cpu().detach().numpy()
    G = MeshDecoder_Output.vertex_channels['G'].cpu().detach().numpy()
    B = MeshDecoder_Output.vertex_channels['B'].cpu().detach().numpy()
    assert len(R) == len(G) == len(B)
    colors = np.vstack((R, G, B)).T
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    Pcd = trimesh.points.PointCloud(vertices=vertices, colors=colors)


    return mesh, Pcd


save_dir = "/remote-home/share/room_gen/shape-output"

models_json = os.path.join(save_dir, 'models-json.json')



model_path = "/remote-home/share/room_gen/room-generation-ckpt/shap-e"
os.makedirs(model_path, exist_ok= True)
pipe = ShapEPipeline.from_pretrained(model_path).to("cuda")
guidance_scale = 15.0




@app.route('/predict', methods=['POST'])
def predict():
    try:
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(models_json):
            with open(models_json, 'r') as f:
                records = json.load(f)
        else:
            records = {}
            with open(models_json, 'w')as f:
                json.dump(records,f, indent=4, sort_keys=True)

        #获取需要的描述词
        data = request.get_json()
        prompt = data['prompt']

        latents = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=64,
        #size=256,
        output_type="mesh",
    ).images
        
        render_mode = 'nerf' # you can change this to 'stf'
        size = 64 # this is the size of the renders; higher values take longer to render.

        # Example of saving the latents as meshes.


        MeshDecoder_Output = latents[0]

        mesh, Pcd = MeshDecoder_2_mesh(MeshDecoder_Output)
        mesh.show()
        now = datetime.now()
        time_id = now.strftime("%m-%d-%H-%M-%S")
        current_dir = os.path.join(save_dir,time_id)
        os.makedirs(current_dir)
        
        data['id'] = time_id
        data['path'] = current_dir
    
        records[time_id] = data
        with open(models_json, 'w') as f:
            json.dump(records, f, indent=4, sort_keys=True)
        
        mesh.export(os.path.join(current_dir, 'mesh.obj'))
        Pcd.export(os.path.join(current_dir, 'point-cloud.ply'))
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=64111)  # 使其在公共IP上运行，端口64110




