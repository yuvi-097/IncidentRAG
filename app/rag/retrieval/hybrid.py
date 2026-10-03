"""Hybrid retrieval: run several retrievers with the same query and filters, then fuse
their rankings into one list (see ``fusion``).

Each component returns up to ``depth`` results (more than the final ``top_k``), so
a chunk ranked moderately by both retrievers can still surface after fusion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.retrieval.base import MAX_TOP_K, RetrievedChunk, Retriever
from app.rag.retrieval.fusion import fuse
from app.rag.store import ChunkFilter
from app.schemas.enums import FusionMethod


@dataclass(frozen=True)
class WeightedRetriever:
    retriever: Retriever
    weight: float = 1.0


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(
        self,
        components: Sequence[WeightedRetriever],
        method: FusionMethod = FusionMethod.RRF,
        rrf_k: int = 60,
        depth: int = 50,
    ) -> None:
        names = [c.retriever.name for c in components]
        if not components:
            raise ValueError("a hybrid retriever needs at least one component")
        if len(set(names)) != len(names):
            raise ValueError(f"component names must be unique: {names}")
        if any(c.weight < 0 for c in components) or not any(c.weight > 0 for c in components):
            raise ValueError("weights must be >= 0 and at least one must be > 0")
        if not 1 <= depth <= MAX_TOP_K:
            raise ValueError(f"depth must be between 1 and {MAX_TOP_K}")
        self.components = list(components)
        self.method = FusionMethod(method)
        self.rrf_k = rrf_k
        self.depth = depth

    @property
    def weights(self) -> dict[str, float]:
        return {c.retriever.name: c.weight for c in self.components}

    def search(
        self, query: str, top_k: int = 10, filters: ChunkFilter | None = None
    ) -> list[RetrievedChunk]:
        query = self.validate(query, top_k)
        depth = max(self.depth, top_k)
        rankings: dict[str, list[tuple[str, float]]] = {}
        chunks: dict[str, RetrievedChunk] = {}
        matched: dict[str, dict[str, None]] = {}
        for component in self.components:
            if component.weight == 0:  # disabled: not even queried
                continue
            results = component.retriever.search(query, depth, filters)
            rankings[component.retriever.name] = [(r.chunk_id, r.score) for r in results]
            for result in results:
                chunks.setdefault(result.chunk_id, result)
                matched.setdefault(result.chunk_id, {}).update(dict.fromkeys(result.matched_terms))
        fused = fuse(self.method, rankings, self.weights, self.rrf_k)[:top_k]
        output = []
        for rank, hit in enumerate(fused, 1):
            details: dict[str, float] = {"fused_score": round(hit.score, 6)}
            for name in rankings:
                if name in hit.ranks:
                    details[f"{name}_rank"] = hit.ranks[name]
                    details[f"{name}_score"] = round(hit.scores[name], 6)
            output.append(
                chunks[hit.chunk_id].model_copy(
                    update={
                        "score": round(hit.score, 6),
                        "rank": rank,
                        "retriever": self.name,
                        "score_details": details,
                        "matched_terms": tuple(matched[hit.chunk_id]),
                    }
                )
            )
        return output
