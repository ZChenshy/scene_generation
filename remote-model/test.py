import requests
import json
import os


def prompt_generate_test_process(data):
    """
    必须确保传入的data有category
    data = {
        "jid": "********"
        "category": "chair",
        "style": "classical",
        "material": "wood"
    }
    如果没有category，就会选择使用caption
    """
    # API 的 URL
    url = "http://183.195.182.126:21258/predict"

    # 发送 POST 请求
    response = requests.post(url, json=data)
    response_json = response.json()  # 解析 JSON 数据
    prompt = response_json.get('prompt', '')
    prompt = prompt.split("<|assistant|>\n")[-1]
    result = response_json
    result['prompt'] =prompt + "It is absolutely imperative to ensure that the top surface of the generated furniture faces upwards."
    return result
    

def shape_test_process(data):
    """
    必须确保data中包含prompt
    data = {
         "prompt": "a bed",
         }
    """
    # API 的 URL
    url = "http://183.195.182.126:21259/predict"

    # 发送 POST 请求
    response = requests.post(url, json=data)
    response_json = response.json()  # 解析 JSON 数据
    
    return response_json


if __name__ =="__main__":
    scene_path = "/remote-home/share/room_gen/3D-FRONT-parsed/"
    capation_path = "/remote-home/share/room_gen/caption_json"
    scene_id = "4944051f-3a7e-4387-b5f3-f925ae6da57e"
    room_id = "LivingRoom-4719"

    with open(os.path.join(scene_path,scene_id+".json"), 'r') as f:
        scene_json = json.load(f)

    
    record = {}
    for room_json in scene_json:
        for furniture in room_json["room_furnitures"]:
            data = furniture
            data["room_instance_id"] = room_json["room_instance_id"]
            
            if data['category'] is not None:
                if data['category'] in record:
                    furniture["id"] = record[data['category']][0]
                    furniture["path"] = record[data['category']][1]
                else:
                    result = prompt_generate_test_process(data)
                    result = shape_test_process(result)
                    print(result)
                    record[result["category"]] = [result["id"], result["path"]]
                    furniture["id"] = result["id"]
                    furniture["path"] = result["path"]
            else:
                result = prompt_generate_test_process(data)
                result = shape_test_process(result)
                print(result)
                record[result["category"]] = [result["id"], result["path"]]
                furniture["id"] = result["id"]
                furniture["path"] = result["path"]
    with open(os.path.join(scene_path,scene_id+".json"), 'w') as f:
        json.dump(scene_json, f, indent=4)

    for room in scene_json:
        print(room)