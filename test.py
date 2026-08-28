import os
import base64
import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv

load_dotenv()

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ====== 改成你电脑上一张真实存在的图片路径 ======
local_image_path = "C:/Users/zhigu/Desktop/软工大作业/代码集/图片集/蒙.jpg"
# =================================================

try:
    print("正在调用 Qwen-VL (DashScope 原生方式，使用本地图片)...")
    
    with open(local_image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{image_base64}"},
                {"text": "请分析这幅画的内容，包括描述、创作技巧和情感表达。"}
            ]
        }
    ]
    
    response = MultiModalConversation.call(
        model="qwen-vl-max",
        messages=messages
    )
    
    if response.status_code == 200:
        print("✅ API 调用成功！")
        
        # ========== 修复后的解析逻辑 ==========
        # 从返回的 content 列表中提取文本
        content_list = response.output.choices[0].message.content
        for item in content_list:
            if "text" in item:  # 检查每个元素是否包含 'text' 键
                print("返回结果:", item["text"])
                break
        # =================================
    else:
        print(f"❌ API 调用失败，状态码: {response.status_code}")
        print("错误信息:", response.message)
        
except FileNotFoundError:
    print(f"❌ 错误：找不到图片文件，请检查路径: {local_image_path}")
except Exception as e:
    print("❌ 发生异常:", str(e))