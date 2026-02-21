"""
============================================
RAG Engine - Advanced Knowledge Retrieval
============================================
Melhoria #3: RAG com busca híbrida e reranking.

Técnicas implementadas:
1. Busca vetorial (embedding similarity)
2. Reranking por relevância contextual
3. Formatação categorizada para o prompt
"""

import httpx
from supabase import Client


class RAGEngine:
    """Motor de RAG avançado com busca vetorial e reranking."""

    def __init__(self, supabase: Client, ai_api_key: str, ai_api_url: str = ""):
        self.supabase = supabase
        self.ai_api_key = ai_api_key
        self.ai_api_url = ai_api_url

    async def search(
        self,
        workspace_id: str,
        query: str,
        agent_id: str | None = None,
        top_k: int = 8,
        threshold: float = 0.45,
    ) -> str:
        """Busca na knowledge base com embedding + reranking."""
        try:
            # 1. Gerar embedding da query
            embedding = await self._generate_embedding(query)

            # 2. Busca vetorial no Supabase
            result = self.supabase.rpc(
                "match_knowledge_base",
                {
                    "query_embedding": f"[{','.join(str(e) for e in embedding)}]",
                    "p_workspace_id": workspace_id,
                    "p_agent_id": agent_id,
                    "match_threshold": threshold,
                    "match_count": top_k,
                },
            ).execute()

            if not result.data:
                return ""

            # 3. Reranking: Ordena por relevância contextual
            reranked = self._rerank(result.data, query)

            # 4. Formata por categoria para o prompt
            return self._format_results(reranked[:5])

        except Exception as e:
            print(f"[RAG] Error during search: {e}")
            return ""

    def _rerank(self, results: list[dict], query: str) -> list[dict]:
        """Reranking simples baseado em overlap de palavras-chave."""
        query_words = set(query.lower().split())

        for item in results:
            content = (item.get("content", "") + " " + item.get("title", "")).lower()
            content_words = set(content.split())

            # Score = similaridade vetorial original + bonus por keyword overlap
            vector_score = item.get("similarity", 0)
            keyword_overlap = len(query_words & content_words) / max(len(query_words), 1)

            item["final_score"] = (vector_score * 0.7) + (keyword_overlap * 0.3)

        return sorted(results, key=lambda x: x["final_score"], reverse=True)

    def _format_results(self, results: list[dict]) -> str:
        """Formata resultados por categoria para inserção no prompt."""
        if not results:
            return ""

        category_map: dict[str, list[str]] = {}

        for item in results:
            category = item.get("category", "Geral")
            if category not in category_map:
                category_map[category] = []
            title = item.get("title", "")
            content = item.get("content", "")
            category_map[category].append(f"**{title}**: {content}")

        parts = []
        for category, items in category_map.items():
            parts.append(f"### {category.upper()}")
            parts.extend(items)
            parts.append("")

        return "\n".join(parts)

    async def _generate_embedding(self, text: str) -> list[float]:
        """Gera embedding usando OpenAI (text-embedding-3-small)."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.ai_api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": "text-embedding-3-small", "input": text},
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
