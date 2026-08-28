#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 效果评测脚本
对比 纯VLM 和 VLM+RAG 的回答准确率
"""

import os
import json
import time
from typing import Dict, List, Tuple

# 导入你的模块
from rag_module import search_knowledge
import dashscope
from dashscope import MultiModalConversation, Generation
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


class RAGEvaluator:
    """RAG效果评估器"""

    def __init__(self):
        self.results = []

    def call_vlm(self, image_path: str, question: str) -> str:
        """仅调用视觉模型（无RAG）"""
        try:
            import base64
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{image_base64}"},
                        {"text": f"请回答这个问题：{question} 请简洁回答，不超过50个字。"}
                    ]
                }
            ]

            response = MultiModalConversation.call(
                model="qwen3.7-flash",
                messages=messages
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                for item in content:
                    if "text" in item:
                        return item["text"]
            return "无法回答"
        except Exception as e:
            print(f"VLM调用失败: {e}")
            return "调用失败"

    def call_vlm_with_rag(self, image_path: str, question: str) -> str:
        """调用视觉模型 + RAG检索"""
        try:
            import base64
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            # 1. RAG检索
            rag_results = search_knowledge(question, top_k=3)
            rag_context = ""
            if rag_results:
                rag_context = "\n\n【参考知识】\n" + "\n---\n".join([
                    f"{r['content']}" for r in rag_results
                ])

            # 2. 构建带RAG的prompt
            system_prompt = f"""你是一位艺术鉴赏专家。请基于图片分析和参考知识回答用户问题。

参考知识：{rag_context}

请简洁回答，不超过50个字。"""

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{image_base64}"},
                        {"text": question}
                    ]
                }
            ]

            response = MultiModalConversation.call(
                model="qwen3.7-flash",
                messages=messages
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                for item in content:
                    if "text" in item:
                        return item["text"]
            return "无法回答"
        except Exception as e:
            print(f"VLM+RAG调用失败: {e}")
            return "调用失败"

    def evaluate(self, test_cases: List[Dict]) -> Dict:
        """运行评测"""
        results = []

        for i, case in enumerate(test_cases):
            print(f"\n 测试 {i+1}/{len(test_cases)}: {case['question']}")

            # 基线：纯VLM
            print("  纯VLM...")
            answer_baseline = self.call_vlm(case['image_path'], case['question'])

            # RAG增强
            print("  VLM+RAG...")
            answer_rag = self.call_vlm_with_rag(case['image_path'], case['question'])

            results.append({
                'id': i + 1,
                'question': case['question'],
                'expected': case['expected'],
                'baseline': answer_baseline,
                'rag': answer_rag
            })

            print(f"  期望: {case['expected']}")
            print(f"  纯VLM: {answer_baseline}")
            print(f"  VLM+RAG: {answer_rag}")

            time.sleep(0.5)  # 避免请求过快

        # 计算准确率
        baseline_correct = sum(1 for r in results if r['expected'].lower() in r['baseline'].lower() or r['baseline'].lower() in r['expected'].lower())
        rag_correct = sum(1 for r in results if r['expected'].lower() in r['rag'].lower() or r['rag'].lower() in r['expected'].lower())

        total = len(results)

        return {
            'total': total,
            'baseline_correct': baseline_correct,
            'baseline_accuracy': baseline_correct / total * 100,
            'rag_correct': rag_correct,
            'rag_accuracy': rag_correct / total * 100,
            'improvement': (rag_correct - baseline_correct) / total * 100,
            'details': results
        }


# ========== 主程序 ==========
if __name__ == "__main__":
    # 准备测试数据
    test_cases = [
        # 达芬奇
        {
            'image_path': '图片集/蒙娜丽莎.jpg',
            'question': '达芬奇在《维特鲁威人》中体现了什么思想？？',
            'expected': '人是万物的尺度'
        },
        # ===== 梵高 =====
        {
            'image_path': '图片集/星月夜.jpg',
            'question': '梵高一生创作了多少幅作品？',
            'expected': '约2100幅'
        },
        {
            'image_path': '图片集/星月夜.jpg',
            'question': '《星月夜》是在什么地方创作的？',
            'expected': '圣雷米精神病院'
        },
        {
            'image_path': '图片集/星月夜.jpg',
            'question': '梵高的《向日葵》是为谁创作的？',
            'expected': '高更'
        },
        {
            'image_path': '图片集/星月夜.jpg',
            'question': '梵高为什么在生前只卖出一幅画？',
            'expected': '不被认可'
        },
        {
            'image_path': '图片集/星月夜.jpg',
            'question': '梵高去世时多少岁？',
            'expected': '37岁'
        },
        # ===== 莫奈 =====
        {
            'image_path': '图片集/印象日出.jpg',
            'question': '莫奈为什么被称为"印象派之父"？',
            'expected': '创始人'
        },
        {
            'image_path': '图片集/印象日出.jpg',
            'question': '《印象·日出》是在哪里创作的？',
            'expected': '勒阿弗尔港口'
        },
        {
            'image_path': '图片集/印象日出.jpg',
            'question': '莫奈的《睡莲》系列创作于哪个花园？',
            'expected': '吉维尼花园'
        },
        {
            'image_path': '图片集/印象日出.jpg',
            'question': '"印象派"这个名称是怎么来的？',
            'expected': '被批评家嘲讽'
        },
        {
            'image_path': '图片集/印象日出.jpg',
            'question': '莫奈创作了多少幅《鲁昂大教堂》系列？',
            'expected': '约30幅'
        },

        # ===== 齐白石 =====
        {
            'image_path': '图片集/虾.jpg',
            'question': '齐白石早年从事什么职业？',
            'expected': '木匠'
        },
        {
            'image_path': '图片集/虾.jpg',
            'question': '齐白石提倡什么艺术理念？',
            'expected': '似与不似之间'
        },
        {
            'image_path': '图片集/蛙声十里出山泉.jpg',
            'question': '《蛙声十里出山泉》是为谁创作的？',
            'expected': '老舍'
        },
        {
            'image_path': '图片集/虾.jpg',
            'question': '齐白石为什么被誉为"人民艺术家"？',
            'expected': '艺术成就'
        },
        {
            'image_path': '图片集/虾.jpg',
            'question': '齐白石原名是什么？',
            'expected': '齐纯芝'
        },

        # ===== 张择端 =====
        {
            'image_path': '图片集/清明上河图.jpg',
            'question': '《清明上河图》中画了多少个人物？',
            'expected': '800多人'
        },
        {
            'image_path': '图片集/清明上河图.jpg',
            'question': '《清明上河图》全长多少？',
            'expected': '5.28米'
        },
        {
            'image_path': '图片集/清明上河图.jpg',
            'question': '《清明上河图》采用了什么透视法？',
            'expected': '散点透视'
        },
        {
            'image_path': '图片集/清明上河图.jpg',
            'question': '张择端师从哪位画家？',
            'expected': '李公麟'
        },
        {
            'image_path': '图片集/清明上河图.jpg',
            'question': '明代董其昌如何评价《清明上河图》？',
            'expected': '天下第一神品'
        },
    ]

    print("=" * 60)
    print("开始 RAG 效果评测")
    print("=" * 60)

    evaluator = RAGEvaluator()
    result = evaluator.evaluate(test_cases)

    print("\n" + "=" * 60)
    print("评测结果")
    print("=" * 60)
    print(f"总测试数: {result['total']}")
    print(
        f"纯VLM 正确数: {result['baseline_correct']}/{result['total']} 准确率: {result['baseline_accuracy']:.1f}%"
    )
    print(
        f"VLM+RAG 正确数: {result['rag_correct']}/{result['total']} 准确率: {result['rag_accuracy']:.1f}%"
    )
    print(f"📈 提升幅度: +{result['improvement']:.1f}%")

    # 保存结果
    with open('evaluation_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n 结果已保存到 evaluation_result.json")
