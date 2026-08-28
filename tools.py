import os
import json
from typing import Optional

# 从你的现有模块导入
from rag_module import search_knowledge as rag_search
from vl_api import call_qwen_vl_api

def artwork_analysis(image_path: str) -> str:
    """工具1：分析图片中的艺术作品（调用 Qwen-VL）"""
    try:
        result = call_qwen_vl_api(image_path)
        # 把返回的字典转成可读的文本描述
        summary = f"""
        作品名称：{result.get('title', '未知')}
        艺术家：{result.get('artist', '未知')}
        创作年代：{result.get('year', '未知')}
        艺术风格：{result.get('style', '未知')}
        作品描述：{result.get('description', '')[:150]}...
        构图分析：{result.get('composition', '未知')}
        色彩分析：{result.get('color', '未知')}
        情感表达：{result.get('emotion', '未知')}
        """
        return summary.strip()
    except Exception as e:
        return f"图片分析失败：{str(e)}"

def knowledge_search(query: str) -> str:
    """工具2：从知识库检索相关信息（调用 RAG）"""
    try:
        results = rag_search(query, top_k=3)
        if not results:
            return "未找到相关信息"
        # 把结果拼接成易读的文本
        parts = []
        for r in results:
            parts.append(f"【来源】{r['source']}\n{r['content']}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"知识库检索失败：{str(e)}"

def museum_search(artist: str = None, artwork: str = None, museum_name: str = None, **kwargs) -> str:
    """工具3：查询博物馆信息"""
    # ========== 关键修改：处理额外的参数 ==========
    # 如果传入了额外的参数，尝试从中提取有用的信息
    if kwargs:
        # 常见的参数名列表
        possible_names = ['query', 'name', 'museum_name', 'q', 'search', 'keyword']
        for key in possible_names:
            if key in kwargs and kwargs[key]:
                museum_name = kwargs[key]
                break
        # 如果还没有，就取第一个非空的值
        if not museum_name:
            for value in kwargs.values():
                if value:
                    museum_name = str(value)
                    break
    # ==============================================

    # 处理 museum_name 参数
    if museum_name:
        museum_data = {
            "卢浮宫": "巴黎卢浮宫博物馆",
            "巴黎卢浮宫博物馆": "巴黎卢浮宫博物馆",
            "蒙娜丽莎": "收藏于巴黎卢浮宫博物馆",
            "Mona Lisa": "收藏于巴黎卢浮宫博物馆",
            "大英博物馆": "大英博物馆",
            "故宫博物院": "故宫博物院",
            "纽约现代艺术博物馆": "MoMA",
            "梵高博物馆": "阿姆斯特丹梵高博物馆",
            "北京故宫博物院": "故宫博物院"
        }
        # 尝试直接匹配
        for key, value in museum_data.items():
            if museum_name.lower() in key.lower() or key.lower() in museum_name.lower():
                return f"{museum_name} 的相关信息：{value}"
        # 如果没找到精确匹配，返回一个通用信息
        return f"关于 {museum_name}：这是著名的艺术博物馆/作品，收藏于世界重要艺术机构。"
    
    # 原有的 artist/artwork 匹配逻辑
    museum_data = {
        "达芬奇": {"蒙娜丽莎": "巴黎卢浮宫博物馆", "最后的晚餐": "米兰圣玛丽亚感恩修道院", "维特鲁威人": "威尼斯学院美术馆"},
        "梵高": {"星月夜": "纽约现代艺术博物馆", "向日葵": "阿姆斯特丹梵高博物馆"},
        "莫奈": {"印象·日出": "巴黎马蒙坦莫奈博物馆", "睡莲": "巴黎橘园美术馆", "鲁昂大教堂": "多个博物馆"},
        "张择端": {"清明上河图": "北京故宫博物院"},
        "齐白石": {"虾": "北京画院美术馆", "蛙声十里出山泉": "北京画院美术馆"}
    }
    
    if artist and artwork:
        artist_works = museum_data.get(artist, {})
        for work_name, location in artist_works.items():
            if artwork in work_name or work_name in artwork:
                return f"{work_name} 收藏于 {location}"
        return f"未找到 {artist} 的《{artwork}》的收藏信息"

    if artist:
        artist_works = museum_data.get(artist, {})
        if artist_works:
            locations = set(artist_works.values())
            return f"{artist} 的作品主要收藏于：{'、'.join(locations)}"
        return f"未找到 {artist} 的博物馆信息"

    if artwork:
        for artist_name, works in museum_data.items():
            for work_name, location in works.items():
                if artwork in work_name or work_name in artwork:
                    return f"{work_name} 收藏于 {location}"
        return f"未找到《{artwork}》的收藏信息"

    return "请提供艺术家、作品名称或博物馆名称"