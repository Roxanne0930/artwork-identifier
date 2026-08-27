from flask import Flask, request, jsonify
import os
from flask_cors import CORS  
import base64
from openai import OpenAI
import json

app = Flask(__name__)
CORS(app)  

user_sessions = {}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

client = OpenAI(
    api_key="sk-b9fb3ee99af84afda14f15aa21a9bf81",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

if not os.path.exists('uploads'):
    os.makedirs('uploads')

if not os.path.exists('results'):
    os.makedirs('results')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未上传文件', 'status': 400}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '无效文件', 'status': 400}), 400

        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # 调用Qwen-VL模型进行图像识别
        result = call_qwen_vl_api(file_path)

        result_path = os.path.join('results', f"{os.path.splitext(file.filename)[0]}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        user_id = request.remote_addr
        user_sessions[user_id] = [
            {"role": "system", "content": f"这幅艺术作品的分析结果：{result['description']}"}
        ]

        return jsonify({
            'status': 'success',
            'artwork_info': result
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 500}), 500

def call_qwen_vl_api(image_path):
    """调用Qwen-VL模型解析图片"""
    try:
        base64_image = encode_image(image_path)

        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",  
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {"type": "text", "text": "分析这幅艺术作品，包括描述、创作技巧和情感表达等"},
                    ],
                }
            ],
        )

        response_text = completion.choices[0].message.content

        return {
            'description': response_text,
            'artist': '未知',
            'techniques': [],
            'emotion': '未识别情感'
        }

    except Exception as e:
        print(f"Qwen-VL API调用失败: {e}")
        # 调用失败时返回模拟数据（便于前端测试）
        return {
            'description': '测试描述（API调用失败）',
            'artist': '测试作者',
            'techniques': ['模拟技巧'],
            'emotion': '模拟情感'
        }

def call_qwen_text_api(messages):
    """调用通义千问文本API进行对话，传递完整对话历史"""
    try:
        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",  
            messages=messages  
        )

        response_text = completion.choices[0].message.content
        return response_text

    except Exception as e:
        print(f"通义千问API调用失败: {e}")
        # 调用失败时返回模拟数据（便于前端测试）
        return f"测试响应：无法连接到AI服务"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_id = request.remote_addr  
        data = request.get_json()
        user_message = data.get('message')

        if user_id not in user_sessions:
            user_sessions[user_id] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

        user_sessions[user_id].append({"role": "user", "content": user_message})

        ai_response = call_qwen_text_api(user_sessions[user_id])

        user_sessions[user_id].append({"role": "assistant", "content": ai_response})

        return jsonify({
            'message': ai_response
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 500}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)