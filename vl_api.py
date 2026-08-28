import base64
import json
import re
import dashscope
from dashscope import MultiModalConversation
import os
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

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

def call_qwen_vl_api(image_path):
    """调用Qwen-VL模型解析图片，返回结构化JSON"""
    try:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

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

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{image_base64}"},
                    {"text": "请分析这幅艺术作品，按照系统指令中的JSON格式输出。"}
                ]
            }
        ]

        response = MultiModalConversation.call(
            model="qwen3.7-flash",
            messages=messages
        )

        if response.status_code == 200:
            content_list = response.output.choices[0].message.content
            response_text = ""
            for item in content_list:
                if "text" in item:
                    response_text = item["text"]
                    break

            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = json.loads(response_text)

                required_fields = ['title', 'artist', 'year', 'style', 'description',
                                 'composition', 'color', 'techniques', 'emotion', 'cultural_context']
                for field in required_fields:
                    if field not in result:
                        result[field] = "未知" if field != 'techniques' else []
                return result

            except json.JSONDecodeError:
                print("JSON解析失败，原始返回:", response_text[:200])
                return get_default_result()
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            print("错误信息:", response.message)
            return get_default_result()

    except Exception as e:
        print(f" Qwen-VL API调用失败: {e}")
        return get_default_result()