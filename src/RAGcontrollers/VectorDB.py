import google.genai as genai
from cachetools import LRUCache
from qdrant_client import QdrantClient
from qdrant_client.http import models
from helpers.config import get_settings
from typing import List, Dict, Any
import time
import asyncio


class VectorDB:

    COMMON_QUERIES = [
    "رصيدي",
    "كم رصيدي",
    "عايز أعرف رصيدي",
    "أبي أعرف رصيدي",
    "شنو رصيدي",
    "ساعات العمل",
    "كيف أشحن",
    "باقات الانترنت",
    "إلغاء الاشتراك",
    "التحدث مع موظف",
]
    
    def __init__(self):
        # 1. تهيئة الإعدادات والاتصال بـ Qdrant
        self.settings = get_settings()
        self.qdrant_client = QdrantClient(
            host=self.settings.QDRANT_HOST, 
            port=self.settings.QDRANT_PORT,
        )
        
        # 2. تهيئة عميل Gemini للـ Embeddings
        self.genai_client = genai.Client(api_key=self.settings.GEMINI_API_KEY)
        self.model_name = self.settings.EMBEDDING_MODEL
        
        # 3. التأكد من وجود الـ Collection
        self._ensure_collection()
        self._embed_cache: LRUCache = LRUCache(maxsize=500)


    def get_by_doc_type(self, doc_type: str, limit: int = 30) -> list[str]:
        from qdrant_client.http import models

        scroll_filter = models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))]
        )

        points, _ = self.qdrant_client.scroll(
            collection_name=self.settings.VECTOR_COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )

        texts = []
        for p in points:
            t = (p.payload or {}).get("text", "")
            if t:
                texts.append(t)
        return texts

    async def prewarm_cache(self):
        """
        Embed common queries at startup so the first real call hits cache.
        With paid API, this batch runs in ~1 API call instead of 10.
        """
        print("🔥 Pre-warming embedding cache...")
        embeddings = self.get_embeddings(self.COMMON_QUERIES, is_query=True)
        for text, emb in zip(self.COMMON_QUERIES, embeddings):
            key = self._cache_key(text)
            self._embed_cache[key] = emb
        print(f"   ✓ {len(embeddings)} queries pre-warmed")

    def _ensure_collection(self):
        """تجهيز الـ Collection والتأكد من توافق الأبعاد."""
        expected_dim = int(self.settings.EMBEDDING_DIMENSION)
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == self.settings.VECTOR_COLLECTION_NAME for c in collections)
            
            if exists:
                collection_info = self.qdrant_client.get_collection(
                    collection_name=self.settings.VECTOR_COLLECTION_NAME
                )
                current_dim = collection_info.config.params.vectors.size
                if current_dim != expected_dim:
                    print(f"⚠️ Dimension mismatch: {current_dim} vs {expected_dim}. Recreating...")
                    self.qdrant_client.delete_collection(self.settings.VECTOR_COLLECTION_NAME)
                    exists = False
            
            if not exists:
                self.qdrant_client.create_collection(
                    collection_name=self.settings.VECTOR_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=expected_dim,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"✅ Collection '{self.settings.VECTOR_COLLECTION_NAME}' Ready (Dim: {expected_dim})")
        except Exception as e:
            print(f"❌ Error initializing Qdrant: {e}")

    def _cache_key(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.strip().lower().encode()).hexdigest()
    
    def get_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        # Cache applies only to single query lookups, not batch document inserts
        if is_query and len(texts) == 1:
            key = self._cache_key(texts[0])
            if key in self._embed_cache:
                return [self._embed_cache[key]]

        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        attempts = 3
        backoff = 1.0
        
        for attempt in range(1, attempts + 1):
            try:
                result = self.genai_client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config={'task_type': task_type}
                )

                embeddings = []
                # معالجة استجابة الـ SDK الجديد لضمان الحصول على المصفوفة الصحيحة
                if hasattr(result, 'embeddings'):
                    for emb in result.embeddings:
                        embeddings.append(list(emb.values))
                elif hasattr(result, 'embedding'):
                    embeddings.append(list(result.embedding.values))
                
                if is_query and len(texts) == 1 and embeddings:
                    key = self._cache_key(texts[0])
                    self._embed_cache[key] = embeddings[0]

                return embeddings

            except Exception as e:
                if '429' in str(e) or 'rate' in str(e).lower():
                    if attempt < attempts:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                print(f"❌ Gemini Embedding Error: {e}")
                return []

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """إدخال البيانات في Qdrant."""
        if not chunks: return 0
            
        texts = [c['text'] for c in chunks]
        embeddings = self.get_embeddings(texts, is_query=False)

        if not embeddings: return 0

        points = []
        for i, chunk in enumerate(chunks):
            points.append(models.PointStruct(
                id=chunk['id'],
                vector=embeddings[i],
                payload={
                    "text": chunk["text"],
                    "source": chunk["metadata"].get("source_file"),
                    "chunk_index": chunk["metadata"].get("chunk_index"),
                    "dialect": chunk["metadata"].get("dialect", "all"),

                    # ✅ add these
                    "doc_type": chunk["metadata"].get("doc_type"),
                    "doc_id": chunk["metadata"].get("doc_id"),
                    "source_stored": chunk["metadata"].get("source_stored"),
                    "file_path": chunk["metadata"].get("file_path"),
                    }
            ))

        self.qdrant_client.upsert(
            collection_name=self.settings.VECTOR_COLLECTION_NAME,
            points=points
        )
        return len(points)

    def search(self, query_text: str, limit: int = 5, dialect: str = None) -> List[str]:
        """البحث وإرجاع قائمة نصوص صافية."""
        try:
            # 1. الحصول على الـ Vector للسؤال
            embeddings = self.get_embeddings([query_text], is_query=True)
            if not embeddings:
                return []
            query_vector = embeddings[0]

            # 2. محاولة البحث بأسلوبين لضمان التوافق مع إصدار المكتبة
            try:
                query_filter = None
                if dialect and dialect != "msa":
                    query_filter = models.Filter(
                        should=[
                            models.FieldCondition(key="dialect", match=models.MatchValue(value=dialect)),
                            models.FieldCondition(key="dialect", match=models.MatchValue(value="all")),
                        ]
                    )
                # المحاولة الأولى: استخدام search التقليدية
                search_result = self.qdrant_client.search(
                    collection_name=self.settings.VECTOR_COLLECTION_NAME,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True
                )

                
            except (AttributeError, Exception):
                # المحاولة الثانية: إذا كان الإصدار حديثاً جداً ويستخدم query_points
                print("🔄 Switching search mode (Legacy/New API)...")
                response = self.qdrant_client.query_points(
                    collection_name=self.settings.VECTOR_COLLECTION_NAME,
                    query=query_vector,
                    limit=limit
                )
                search_result = response.points

            # 3. استخراج النصوص من الـ payload
            extracted_texts = []
            for point in search_result:
                payload = point.payload if hasattr(point, 'payload') else {}
                text = payload.get('text', '')
                if text:
                    extracted_texts.append(str(text))

            if extracted_texts:
                print(f"✅ RAG: Found {len(extracted_texts)} segments.")
            
            return extracted_texts

        except Exception as e:
            print(f"❌ Critical Search Error: {e}")
            return []
        
    async def search_async(self, query_text: str, limit: int = 5) -> list:
        return await asyncio.to_thread(self.search, query_text, limit)
        

vector_db = VectorDB()