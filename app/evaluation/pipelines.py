"""The six systems compared in the ablation study (Phase 10).

A. Dense          -- retrieve with dense embeddings, answer from the top 5 chunks
B. BM25           -- same, BM25
C. Hybrid         -- same, hybrid (reciprocal rank fusion of dense and BM25)
D. Hybrid+rerank  -- same, hybrid then the cross-encoder reranker
E. D + verification -- D's context, then the Phase 7 layer: evidence validation
                     (may decline), claim verification (unsupported claims removed),
                     computed confidence, source-conflict notes
F. Full agent     -- the production agent: routing, tools and plans (SQL, temporal,
                     multi-hop), security screening, verification, output validation

What is held constant: access control (every retriever query is filtered to the
caller's grants: it is part of the data layer, not a component to ablate), the
extractive synthesizer (no LLM is configured), the data and the clock. A-D answer
from the top 5 chunks with the same extractive synthesizer the agent uses for
passages, and never decline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from app.agents.entities import extract_entities
from app.agents.evidence import _from_passage
from app.agents.graph import Agent
from app.agents.state import AgentState, EvidenceStatus
from app.agents.synthesis import ExtractiveSynthesizer
from app.evaluation.eval_set import EvalQuestion
from app.evaluation.metrics import ContextItem, context_item
from app.rag.retrieval.base import RetrievedChunk, Retriever
from app.schemas.enums import QueryType
from app.security import principal_for_role
from app.security.principal import Principal
from app.tools.base import Evidence
from app.tools.common import chunk_filter_for

METHODS: dict[str, str] = {
    "A": "Dense",
    "B": "BM25",
    "C": "Hybrid",
    "D": "Hybrid + reranker",
    "E": "Hybrid + reranker + verification",
    "F": "Full agentic system",
}
CONTEXT_K = 5  # chunks given to the reader in A-E
RANK_DEPTH = 10  # chunks scored for retrieval metrics
CANDIDATES = 30  # hybrid candidates the reranker sees (for reranking-error analysis)


class MethodOutput(BaseModel):
    method: str
    question_id: str
    answer: str
    ranked: list[str]  # source record of each ranked item, best first
    candidates: list[str]  # before reranking (for reranking-failure analysis)
    context: list[ContextItem]  # what the answer was written from
    declined: bool = False  # the system itself declined (no-answer, refusal)
    query_type: str | None = None
    plan: str | None = None
    tools: list[str] = []
    confidence: str | None = None
    removed_claims: int = 0
    security: dict[str, object] = {}
    errors: list[str] = []
    latency_ms: float = 0.0


@dataclass
class Retrievers:
    dense: Retriever | None
    bm25: Retriever
    hybrid: Retriever | None
    hybrid_rerank: Retriever | None

    def for_method(self, method: str) -> Retriever | None:
        return {"A": self.dense, "B": self.bm25, "C": self.hybrid, "D": self.hybrid_rerank}.get(
            method
        )


def _principal(question: EvalQuestion) -> Principal:
    return principal_for_role(f"eval-{question.role}", question.role)


class Runner:
    """Runs one evaluation question through one method."""

    def __init__(
        self,
        retrievers: Retrievers,
        agent: Agent,
        synthesizer: ExtractiveSynthesizer,
        clock: Callable[[], datetime],
        snippet_chars: int = 1500,
    ) -> None:
        self.retrievers = retrievers
        self.agent = agent
        self.synthesizer = synthesizer
        self.clock = clock
        self.snippet_chars = snippet_chars
        self._cache: dict[tuple[str, str], list[RetrievedChunk]] = {}

    # --- retrieval -----------------------------------------------------------------------

    def retrieve(self, method: str, question: EvalQuestion, depth: int) -> list[RetrievedChunk]:
        key = (method, question.id)
        cached = self._cache.get(key)
        if cached is None or len(cached) < depth:
            retriever = self.retrievers.for_method(method)
            if retriever is None:
                raise ValueError(f"method {method} is not available in this run")
            principal = _principal(question)
            cached = retriever.search(question.question, depth, chunk_filter_for(principal))
            self._cache[key] = cached
        return cached[:depth]

    def _state(self, question: EvalQuestion, chunks: list[RetrievedChunk]) -> AgentState:
        state = AgentState(
            query=question.question, principal=_principal(question), now=self.clock()
        )
        state.entities = extract_entities(question.question, self.agent.router.services, state.now)
        state.query_type = QueryType.DOCUMENT_SEARCH
        items = [
            _from_passage(Evidence.from_chunk(c, self.snippet_chars), "retrieval") for c in chunks
        ]
        state.retrieved_documents = items
        state.reranked_evidence = [
            item.model_copy(update={"label": f"E{n}", "relevance": 1 / n})
            for n, item in enumerate(items, 1)
        ]
        state.goals = {"documents": bool(items)}
        state.evidence_screened = True  # A-E have no screening stage (see the module docstring)
        return state

    @staticmethod
    def _context(state: AgentState) -> list[ContextItem]:
        if state.evidence_package is not None:
            return [
                context_item(e.label, e.source_id, e.title, e.content, e.file_path, e.timestamp)
                for e in state.evidence_package.evidence
            ]
        return [
            context_item(i.label or "", i.source_id, i.title, i.text, i.location, i.timestamp)
            for i in state.reranked_evidence
        ]

    # --- methods -------------------------------------------------------------------------

    def run(self, method: str, question: EvalQuestion) -> MethodOutput:
        started = time.perf_counter()
        output = self._agent(question) if method == "F" else self._rag(method, question)
        output.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return output

    def _rag(self, method: str, question: EvalQuestion) -> MethodOutput:
        base = "D" if method == "E" else method
        chunks = self.retrieve(base, question, RANK_DEPTH)
        candidates: list[str] = []
        if base == "D" and self.retrievers.hybrid is not None:
            candidates = [c.document_id for c in self.retrieve("C", question, CANDIDATES)]
        state = self._state(question, chunks[:CONTEXT_K])
        declined = False
        if method == "E":
            agent = self.agent
            agent.validate_evidence(state)
            agent.package(state)
            agent.synthesize(state)
            agent.verify(state)
            agent.confidence(state)
            declined = state.synthesis_method == "none" or state.evidence_status is (
                EvidenceStatus.INSUFFICIENT
            )
        else:
            answer = self.synthesizer.synthesize(state)
            state.final_answer = answer
        return MethodOutput(
            method=method,
            question_id=question.id,
            answer=state.final_answer,
            ranked=[c.document_id for c in chunks],
            candidates=candidates,
            context=self._context(state),
            declined=declined,
            confidence=state.confidence.value if method == "E" else None,
            removed_claims=sum(c.action == "removed" for c in state.claims),
            errors=[e.code for e in state.errors],
        )

    def _agent(self, question: EvalQuestion) -> MethodOutput:
        state = self.agent.run(question.question, _principal(question))
        security = state.security
        return MethodOutput(
            method="F",
            question_id=question.id,
            answer=state.final_answer,
            ranked=[i.source_id for i in state.reranked_evidence],
            candidates=[i.source_id for i in state.retrieved_documents],
            context=self._context(state),
            declined=state.synthesis_method == "none" or security.blocked,
            query_type=state.query_type.value if state.query_type else None,
            plan=state.plan,
            tools=[r.tool for r in state.tool_results],
            confidence=state.confidence.value,
            removed_claims=sum(c.action == "removed" for c in state.claims),
            security={
                "blocked": security.blocked,
                "quarantined": len(security.quarantined),
                "output_removed": list(security.output_removed),
                "access_violations": security.access_violations,
                "references_redacted": security.references_redacted,
            },
            errors=[e.code for e in state.errors],
        )


__all__ = ["CONTEXT_K", "METHODS", "MethodOutput", "Retrievers", "Runner"]
