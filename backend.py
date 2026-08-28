from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import base64
# from openai import OpenAI
import json
from dotenv import load_dotenv
import dashscope
from dashscope import MultiModalConversation, Generation
from rag_module import search_knowledge
from agent import run_agent

load_dotenv()

app = Flask(__name__)
CORS(app)

user_sessions = {}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
'''
client = OpenAI(
    api_key= os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
'''
# 设置 DashScope API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

if not os.path.exists('uploads'):
    os.makedirs('uploads')

if not os.path.exists('results'):
    os.makedirs('results')

@app.route('/upload', methods=['POST'])
def upload_file():
    global global_image_path
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未上传文件', 'status': 400}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '无效文件', 'status': 400}), 400

        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        global_image_path = file_path
        print(f"图片已保存: {global_image_path}")

        # 调用Qwen-VL模型进行图像识别
        result = call_qwen_vl_api(file_path)

        result_path = os.path.join(
            'results', f"{os.path.splitext(file.filename)[0]}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        user_id = request.remote_addr
        user_sessions[user_id] = [{
            "role":
            "system",
            "content":
            f"这幅艺术作品的分析结果：{result['description']}"
        }]

        return jsonify({'status': 'success', 'artwork_info': result})

    except Exception as e:
        print(f"upload_file 错误: {e}")
        return jsonify({'error': str(e), 'status': 500}), 500


def call_qwen_vl_api(image_path):
    """调用Qwen-VL模型解析图片，返回结构化JSON"""
    try:
        # 读取图片并转为base64
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 定义期望的JSON结构
        system_prompt = """你是一位专业的艺术鉴赏专家。请分析这幅艺术作品，并严格按照以下JSON格式返回结果（只返回JSON，不要有其他文字）：

{
    "title": "作品名称（如果无法确定，根据内容合理推测）",
    "artist": "艺术家姓名（如果无法确定，写'未知'）",
    "year": "创作年份或时期（如：1889年、20世纪初，如果无法确定写'未知'）",
    "style": "艺术风格（如：印象派、立体主义、中国水墨画等）",
    "description": "详细描述画面内容（150-250字）",
    "composition": "构图分析（如：对称构图、黄金分割、对角线构图等）",
    "color": "色彩分析（如：暖色调、冷色调、对比色等）",
    "techniques": ["技法1", "技法2", "技法3"],
    "emotion": "作品传达的情感或氛围（如：宁静、激情、忧郁等）",
    "cultural_context": "文化背景或历史背景（如果无法确定写'未知'）"
}"""

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{image_base64}"},
                    {"text": "请分析这幅艺术作品，按照系统指令中的JSON格式输出。"}
                ]
            }
        ]

        # 调用模型
        response = MultiModalConversation.call(
            model="qwen-vl-max",
            messages=messages
        )

        if response.status_code == 200:
            # 从返回结果中提取文本
            content_list = response.output.choices[0].message.content
            response_text = ""
            for item in content_list:
                if "text" in item:
                    response_text = item["text"]
                    break

            # 尝试解析JSON
            try:
                import re
                # 提取JSON部分（防止模型返回了多余文字）
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = json.loads(response_text)

                # 确保所有必要字段都存在
                required_fields = ['title', 'artist', 'year', 'style', 'description',
                                 'composition', 'color', 'techniques', 'emotion', 'cultural_context']
                for field in required_fields:
                    if field not in result:
                        result[field] = "未知" if field != 'techniques' else []
                return result

            except json.JSONDecodeError:
                print("JSON解析失败，原始返回:", response_text[:200])
                # 返回默认结构
                return {
                    'title': '未知作品',
                    'artist': '未知',
                    'year': '未知',
                    'style': '未知',
                    'description': response_text[:200] if response_text else '无法解析',
                    'composition': '未知',
                    'color': '未知',
                    'techniques': [],
                    'emotion': '未知',
                    'cultural_context': '未知'
                }
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            print("错误信息:", response.message)
            return get_default_result()

    except Exception as e:
        print(f"❌ Qwen-VL API调用失败: {e}")
        return get_default_result()

def get_default_result():
    """返回默认结果（API调用失败时使用）"""
    return {
        'title': '测试作品',
        'artist': '测试作者',
        'year': '2024',
        'style': '测试风格',
        'description': '这是一幅测试用的艺术作品描述。',
        'composition': '对称构图',
        'color': '暖色调',
        'techniques': ['笔触技法', '色彩渐变'],
        'emotion': '宁静',
        'cultural_context': '现代艺术'
    }


'''
def call_qwen_text_api(messages):
    """调用通义千问文本API进行对话（使用DashScope原生方式）"""
    try:
        response = Generation.call(
            model="qwen-plus",  # 可以换成 qwen-turbo（更便宜）或 qwen-max（更强）
            messages=messages
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            print(f"文本API调用失败: {response.message}")
            return "测试响应：无法连接到AI服务"

    except Exception as e:
        print(f"通义千问API调用失败: {e}")
        return "测试响应：无法连接到AI服务"
'''


def call_qwen_text_api(messages, use_rag=True):
    """调用通义千问文本API进行对话（支持RAG增强）"""
    try:
        # 获取用户最后一轮的问题
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # RAG检索
        rag_context = ""
        if use_rag and user_query:
            print(f" 正在检索知识库: {user_query[:50]}...")
            results = search_knowledge(user_query, top_k=3)
            if results:
                rag_context = "\n\n【参考知识】\n" + "\n---\n".join(
                    [f"来源：{r['source']}\n内容：{r['content']}" for r in results])
                print(f" 检索到 {len(results)} 条相关知识")
            else:
                print(" 未检索到相关知识")

        # 构建增强的system prompt
        enhanced_messages = []
        system_found = False
        for msg in messages:
            if msg.get("role") == "system":
                system_found = True
                enhanced_messages.append({
                    "role": "system",
                    "content": msg.get("content", "") + rag_context
                })
            else:
                enhanced_messages.append(msg)

        if not system_found and rag_context:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": f"你是一位艺术鉴赏专家。请基于以下参考知识回答问题。{rag_context}"
            })

        # ========== 修改开始：改用 MultiModalConversation ==========
        # 转换消息格式：MultiModalConversation 要求 content 为字符串
        converted_messages = []
        for msg in enhanced_messages:
            content = msg.get("content", "")
            # 如果 content 是列表（比如包含 image_url），提取文本部分
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = " ".join(text_parts)
            converted_messages.append({
                "role": msg["role"],
                "content": content
            })

        # 使用 MultiModalConversation 调用
        response = MultiModalConversation.call(
            model="qwen3.7-flash",
            messages=converted_messages
        )
        # ========== 修改结束 ==========

        if response.status_code == 200:
            # 解析 MultiModalConversation 的返回结果
            return response.output.choices[0].message.content[0]["text"]
        else:
            print(f"文本API调用失败: {response.message}")
            return "测试响应：无法连接到AI服务"

    except Exception as e:
        print(f"通义千问API调用失败: {e}")
        return "测试响应：无法连接到AI服务"

@app.route('/chat', methods=['POST'])


def chat():
    try:
        user_id = request.remote_addr
        data = request.get_json()
        user_message = data.get('message')
        
        # 获取当前用户的对话历史（保存在 user_sessions 中）
        if user_id not in user_sessions:
            user_sessions[user_id] = []
        
        # 把用户的问题添加到历史中
        user_sessions[user_id].append({"role": "user", "content": user_message})
        
        # 调用 Agent，传入完整的历史和图片路径
        global global_image_path
        response = run_agent(
            user_query=user_message, 
            image_path=global_image_path,
            history=user_sessions[user_id]  # 新增：传入历史
        )
        
        # 把模型的回答添加到历史中
        user_sessions[user_id].append({"role": "assistant", "content": response})
        
        return jsonify({'message': response})
    except Exception as e:
        print(f"chat接口错误: {e}")
        return jsonify({'error': str(e), 'status': 500}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
