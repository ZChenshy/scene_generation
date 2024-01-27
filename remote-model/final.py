import json
import os
import requests
import torch
from transformers import pipeline
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif
import trimesh
import numpy as np
from datetime import datetime


def run(scene_id,scene_path):
    """
    使用shap e 生成家具粗模
    传入的args参数中必须包含scene_path，scene_id
    """

    #加载text生成模型
    Text_model = "/remote-home/share/room_gen/room-generation-ckpt/zephyr-7b-beta"
    Text_pipe = pipeline("text-generation", model=Text_model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    print(f"prompt generate model loads from {Text_model}")
    #加载shapE
    model_path = "/remote-home/share/room_gen/room-generation-ckpt/shap-e"
    os.makedirs(model_path, exist_ok= True)
    ShapE_pipe = ShapEPipeline.from_pretrained(model_path).to("cuda:1")
    
    # 读取场景json
    with open(os.path.join(scene_path, scene_id+".json"), 'r') as f:
        scene_json = json.load(f)

    record = {}
    for room_json in scene_json:
        for furniture in room_json["room_furnitures"]:
            data = furniture
            data["room_instance_id"] = room_json["room_instance_id"]
            
            if data['category'] is not None:
                if data['category'] in record:#已经生成过改家具（有的场景会包含几个一样的家具）
                    furniture["id"] = record[data['category']][0]
                    furniture["path"] = record[data['category']][1]
                else:
                    result = text_generate(data, Text_pipe)
                    result = shapE_process(result, ShapE_pipe)
                    print(result)
                    record[result["category"]] = [result["id"], result["path"]]
                    furniture["id"] = result["id"]
                    furniture["path"] = result["path"]
            else:
                result = text_generate(data, Text_pipe)
                result = shapE_process(result, ShapE_pipe)
                print(result)
                record[result["category"]] = [result["id"], result["path"]]
                furniture["id"] = result["id"]
                furniture["path"] = result["path"]
    with open(os.path.join(scene_path, scene_id+".json"), 'w') as f:
        json.dump(scene_json, f, indent=4)
    print('Done!')

def text_generate(data, pipe):
    """
    给出家具的category，style，material或者caption，生成一段描述该家具的prompt
    """
    

    category = data['category']
    style = data['style']
    material = data['material']
    caption = data['caption']

    if category is not None:
        messages = [
            {
                "role": "system",
                "content": f"Now, generate a detailed description of a piece of \
                            furniture based on the following attributes: \
                            Furniture Type: {category}; Style: {style}; Material: {material}. \
                            The description should very very simply depict the appearance, style, and material, \
                            highlighting its shape. \
                            You must keep the description between 20 to 30 words.",
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": f"Now, generate a detailed description of a piece of \
                            furniture based on the following text:{caption} \
                            The description should very very simply depict the appearance, style, and material, \
                            highlighting its shape. \
                            You must keep the description between 20 to 30 words.",
            },
        ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50)
    generated_text = outputs[0]["generated_text"]
    print(generated_text.split("<|assistant|>\n")[-1])
    data['prompt'] = generated_text.split("<|assistant|>\n")[-1] + "It is absolutely imperative to ensure that the top surface of the generated furniture faces upwards."
    return data


def shapE_process(data, pipe):
    """
    必须确保data中包含prompt
    data = {
         "prompt": "a bed",
         }
    """
    save_dir = "/remote-home/share/room_gen/shape-output"
    models_json = os.path.join(save_dir, 'models-json.json')
    
    
    guidance_scale = 15.0

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(models_json):
        with open(models_json, 'r') as f:
            records = json.load(f)
    else:
        records = {}
        with open(models_json, 'w')as f:
            json.dump(records,f, indent=4, sort_keys=True)


    prompt = data['prompt']
    latents = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=64,
    #size=256,
    output_type="mesh",
    ).images
        
    MeshDecoder_Output = latents[0]
    mesh, Pcd = MeshDecoder_2_mesh(MeshDecoder_Output)
    #! >>>debug>>>
    #mesh.show()
    #! <<<debug<<<

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
    print(f"obj and ply saved in {current_dir}")
    return data

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


if __name__ == "__main__":
    scene_path = "/remote-home/share/room_gen/3D-FRONT-parsed/"
    scene_id = "0a8d471a-2587-458a-9214-586e003e9cf9"
    run(scene_id,scene_path)
