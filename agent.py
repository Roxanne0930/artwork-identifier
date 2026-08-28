import json
import re
from typing import TypedDict, Annotated, Sequence
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, FunctionMessage, AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from tools import artwork_analysis, knowledge_search, museum_search
import dashscope
from dashscope import MultiModalConversation
import os
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ---------- 定义 Agent 状态 ----------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    image_path: str

# ---------- 手动创建工具 ----------
def tool_artwork_analysis_func(image_path: str) -> str:
    """分析上传的艺术作品图片，返回作品的视觉分析结果。"""
    return artwork_analysis(image_path)

def tool_knowledge_search_func(query: str) -> str:
    """从艺术作品知识库中检索信息，如艺术家生平、风格、代表作等。"""
    return knowledge_search(query)

def tool_museum_search_func(artist: str = None, artwork: str = None, museum_name: str = None, **kwargs) -> str:
    """查询艺术品在哪个博物馆可以参观。"""
    return museum_search(artist, artwork, museum_name, **kwargs)

tool_artwork_analysis = StructuredTool.from_function(
    func=tool_artwork_analysis_func,
    name="tool_artwork_analysis",
    description="分析上传的艺术作品图片，返回作品的视觉分析结果。"
)

tool_knowledge_search = StructuredTool.from_function(
    func=tool_knowledge_search_func,
    name="tool_knowledge_search",
    description="从艺术作品知识库中检索信息，如艺术家生平、风格、代表作等。"
)

tool_museum_search = StructuredTool.from_function(
    func=tool_museum_search_func,
    name="tool_museum_search",
    description="查询艺术品在哪个博物馆可以参观。"
)

tools = [tool_artwork_analysis, tool_knowledge_search, tool_museum_search]

# ---------- 解析工具调用（修复版） ----------
def parse_tool_call(content: str):
    """从模型回复中提取工具调用"""
    # 方法1：直接解析整个内容
    try:
        data = json.loads(content.strip())
        if "tool" in data:
            return data
    except:
        pass
    
    # 方法2：提取完整的 JSON 对象（处理嵌套）
    try:
        start = content.find('{')
        if start == -1:
            return None
        brace_count = 0
        for i in range(start, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = content[start:i+1]
                    data = json.loads(json_str)
                    if "tool" in data:
                        return data
                    break
    except:
        pass
    
    # 方法3：用正则提取
    try:
        pattern = r'\{[^{}]*"tool"[^{}]*"args"\s*:\s*\{[^{}]*\}\s*\}'
        match = re.search(pattern, content)
        if match:
            return json.loads(match.group())
    except:
        pass
    
    return None

# ---------- 执行工具 ----------
def execute_tool(tool_name: str, args: dict, image_path: str):
    print(f"🔧 执行工具: {tool_name}, args: {args}")
    
    if tool_name == "tool_artwork_analysis":
        if not image_path:
            return "错误：未提供图片路径，请先上传图片"
        result = tool_artwork_analysis.invoke({"image_path": image_path})
        print(f"   ✅ 图片分析完成，结果长度: {len(result)}")
        return result
    
    elif tool_name == "tool_knowledge_search":
        query = args.get("query", "")
        if not query:
            return "错误：未提供搜索关键词"
        result = tool_knowledge_search.invoke({"query": query})
        print(f"   ✅ 知识检索完成，结果长度: {len(result)}")
        return result
    
    elif tool_name == "tool_museum_search":
        result = tool_museum_search.invoke(args)
        print(f"   ✅ 博物馆查询完成，结果长度: {len(result)}")
        return result
    
    else:
        return f"未知工具：{tool_name}"

# ---------- 调用 LLM ----------
def call_llm(messages: list, image_path: str):
    """调用 Qwen 模型，支持工具调用"""
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, FunctionMessage):
            formatted_messages.append({
                "role": "assistant", 
                "content": f"【工具返回的事实】{msg.content}"
            })
    
    system_prompt = """你是一个艺术作品研究助手。必须遵守以下规则：

1. 如果用户问的是关于当前图片的问题（如"这幅画是谁画的"），你必须先调用 tool_artwork_analysis。
2. **调用工具后，你必须基于工具返回的结果来回答。工具返回的内容是事实，不要自己编造答案。**
3. 如果用户问的是艺术家知识，调用 tool_knowledge_search。
4. 如果用户问的是博物馆信息，调用 tool_museum_search。

调用格式：{"tool": "工具名称", "args": {"参数名": "参数值"}}
注意：tool_artwork_analysis 不需要你传参，系统会自动传入图片路径。

如果不调用工具，直接输出最终回答。
"""
    
    full_messages = [{"role": "system", "content": system_prompt}] + formatted_messages
    
    print(f"📨 发送给模型的消息数: {len(full_messages)}")
    
    response = MultiModalConversation.call(
        model="qwen3.7-flash",
        messages=full_messages,
        result_format="message"
    )
    
    print(f"📡 响应状态码: {response.status_code}")
    
    if response.status_code == 200:
        content_list = response.output.choices[0].message.content
        content_text = ""
        for item in content_list:
            if "text" in item:
                content_text = item["text"]
                break
        print(f"✅ LLM 返回: {content_text[:150]}...")
        return AIMessage(content=content_text)
    else:
        print(f"❌ 模型调用失败: {response.message}")
        return AIMessage(content=f"模型调用失败：{response.message}")

# ---------- 主流程 ----------
def run_agent(user_query: str, image_path: str = None, history: list = None) -> str:
    print(f"🚀 启动 Agent: {user_query}")
    print(f"📁 图片路径: {image_path}")
    
    messages = []
    if history:
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=user_query))
    
    for round_num in range(5):
        print(f"🔄 第 {round_num + 1} 轮")
        
        response_msg = call_llm(messages, image_path)
        messages.append(response_msg)
        
        if not isinstance(response_msg, AIMessage):
            continue
        
        content = response_msg.content
        
        tool_call = parse_tool_call(content)
        if tool_call:
            print("🔄 检测到工具调用，执行工具...")
            tool_name = tool_call.get("tool")
            args = tool_call.get("args", {})
            
            result = execute_tool(tool_name, args, image_path)
            messages.append(FunctionMessage(content=result, name=tool_name))
            continue
        else:
            print("✅ 无工具调用，返回最终回答")
            break
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
        if isinstance(msg, FunctionMessage) and msg.content:
            return f"工具返回：{msg.content}"
    
    return "Agent 未能生成回答"

if __name__ == "__main__":
    test_result = run_agent("达芬奇的作品有什么特点？")
    print(test_result)