import os
import json
import hashlib
from typing import List, Dict, Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class ArtKnowledgeBase:
    """艺术作品知识库 - 基于RAG的检索增强系统"""

    def __init__(self, persist_directory="./chroma_db"):
        """初始化知识库"""
        self.persist_directory = persist_directory
        
        print(" 正在加载嵌入模型...")
        
        # ========== 修改开始 ==========
        # 创建一个包装类，让 ChromaDB 能识别
        from chromadb.api.types import EmbeddingFunction
        
        class HuggingFaceEmbeddingWrapper(EmbeddingFunction):
            def __init__(self, model_name):
                self.model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            def name(self):
                return "huggingface"    
                
            def __call__(self, texts):
                return self.model.embed_documents(texts)
        
        # 使用包装类
        self.embedding_function = HuggingFaceEmbeddingWrapper(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # ========== 修改结束 ==========
        
        print(" 嵌入模型加载完成")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name="art_knowledge",
            embedding_function=self.embedding_function
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

    def load_documents_from_folder(self, folder_path: str):
        """从文件夹加载文档"""
        documents = []

        if not os.path.exists(folder_path):
            print(f" 文件夹 '{folder_path}' 不存在")
            return documents

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not os.path.isfile(file_path):
                continue

            try:
                if filename.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                elif filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                else:
                    continue

                docs = loader.load()
                documents.extend(docs)
                print(f" 加载文档: {filename}")

            except Exception as e:
                print(f" 加载文档 {filename} 失败: {e}")

        return documents

    def chunk_and_index(self, documents: List[Any]):
        """切分文档并建立索引"""
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)

            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:  # 过滤太短的片段
                    continue

                chunk_id = hashlib.md5(
                    f"{doc.metadata.get('source', '')}_{i}_{chunk[:50]}".
                    encode()).hexdigest()[:16]

                all_chunks.append(chunk)
                all_metadatas.append({
                    'source':
                    doc.metadata.get('source', '未知来源'),
                    'chunk_index':
                    i
                })
                all_ids.append(chunk_id)

        if all_chunks:
            # 分批添加（避免一次性添加太多）
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                end = min(i + batch_size, len(all_chunks))
                self.collection.add(documents=all_chunks[i:end],
                                    metadatas=all_metadatas[i:end],
                                    ids=all_ids[i:end])
                print(f" 已索引 {end}/{len(all_chunks)} 个文档片段")

        return len(all_chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关知识"""
        results = self.collection.query(query_texts=[query], n_results=top_k)

        if results['documents'] and len(results['documents']) > 0:
            return [{
                'content':
                results['documents'][0][i],
                'source':
                results['metadatas'][0][i].get('source', '未知来源'),
                'score':
                results['distances'][0][i] if results.get('distances') else 0
            } for i in range(len(results['documents'][0]))]
        return []

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'persist_directory': self.persist_directory
        }


# 单例模式
_knowledge_base = None

def get_knowledge_base() -> ArtKnowledgeBase:
    """获取知识库单例"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = ArtKnowledgeBase()
    return _knowledge_base


def build_knowledge_base(folder_path: str = "./文档集"):
    """构建知识库（主入口函数）"""
    kb = get_knowledge_base()

    print(" 开始构建知识库...")

    # 加载文档
    documents = kb.load_documents_from_folder(folder_path)
    if not documents:
        print(" 未找到任何文档，请检查文件夹路径")
        return 0

    print(f" 共加载 {len(documents)} 个文档")

    # 切分并索引
    chunk_count = kb.chunk_and_index(documents)
    print(f" 知识库构建完成，共 {chunk_count} 个文档片段")

    stats = kb.get_stats()
    print(f" 知识库统计: {stats}")

    return chunk_count


def search_knowledge(query: str, top_k: int = 5) -> List[Dict]:
    """搜索知识库（供外部调用）"""
    kb = get_knowledge_base()
    return kb.search(query, top_k)
