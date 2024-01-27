# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline
from flask import Flask, request, jsonify
app = Flask(__name__)

model_path = "/remote-home/share/room_gen/room-generation-ckpt/zephyr-7b-beta"
torch.set_default_device("cuda")
pipe = pipeline("text-generation", model=model_path, torch_dtype=torch.bfloat16, device_map="auto")
print(f"model loads from {model_path}")
# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating

furniture_type = "table"
style = "classical"
material = "wood"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        #获取需要的描述词
        data = request.get_json()
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
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        generated_text = outputs[0]["generated_text"]
        data['prompt'] = generated_text

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=64110)  # 使其在公共IP上运行，端口64110
