#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识库构建脚本 - 一次性运行
将 文档集/knowledge_base/ 下的文档切分并存入向量数据库
"""

from rag_module import build_knowledge_base
import os


if __name__ == "__main__":
    # 检查文档集文件夹
    doc_folder = "./knowledge_base"
    
    if not os.path.exists(doc_folder):
        print(f"文件夹 '{doc_folder}' 不存在")
        print("请确保 '/knowledge_base/' 文件夹存在，并包含 .txt 知识文档")
    else:
        print(f"正在从 '{doc_folder}' 构建知识库...")
        
        # 列出文件夹内容
        files = os.listdir(doc_folder)
        txt_files = [f for f in files if f.endswith('.txt')]
        
        if not txt_files:
            print(" 未找到任何 .txt 文件，请先创建知识库文档")
            print(" 需要包含以下文档：达芬奇.txt、梵高.txt、张择端.txt、莫奈.txt、齐白石.txt")
        else:
            print(f" 找到 {len(txt_files)} 个文本文件: {', '.join(txt_files)}")
            
            # 构建知识库
            count = build_knowledge_base(doc_folder)
            
            if count > 0:
                print(f"\n构建完成！共 {count} 个文档片段")
                print("知识库已保存到 ./chroma_db 文件夹")
                print("\n 现在可以启动后端测试 RAG 效果了:")
                print("   python 1-backend.py")
            else:
                print("\n 构建失败，请检查文档格式是否正确")