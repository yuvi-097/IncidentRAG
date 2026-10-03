"""Build the agent from configuration: retrieval mode, reranker, LLM, tools."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.engine import Engine

from app.agents.graph import Agent, build_router
from app.agents.synthesis import ExtractiveSynthesizer, LLMSynthesizer
from app.agents.verification import load_nli
from app.config import Settings
from app.database.session import create_sql_reader_engine
from app.llm import build_llm
from app.rag.reranking.cross_encoder import CrossEncoderReranker
from app.rag.retrieval.factory import RetrievalComponents
from app.rag.retrieval.pipeline import TimedRetriever
from app.security import load_policy
from app.security.injection_model import load_semantic_detector
from app.tools.registry import build_registry

logger = logging.getLogger(__name__)


def build_agent(
    engine: Engine,
    settings: Settings,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    components: RetrievalComponents | None = None,
    use_sql_reader: bool = True,
) -> Agent:
    """The production agent. ``components`` shares already-loaded retrieval models (the
    evaluation builds them once for every method); ``use_sql_reader=False`` runs SQL on
    ``engine`` behind the other safety layers (for the SQLite evaluation database)."""
    components = components or RetrievalComponents(engine, settings)
    retriever = TimedRetriever(components.retriever(settings.retrieval.mode))
    scorer = None
    if settings.reranker.enabled and settings.agent.rerank_evidence:
        reranker = components.reranking()
        scorer = reranker.scorer if isinstance(reranker, CrossEncoderReranker) else None
    llm = build_llm(settings.llm)
    extractive = ExtractiveSynthesizer(components.bm25.index)
    synthesizer = LLMSynthesizer(llm, extractive) if llm else extractive
    nli = load_nli(settings.verification)
    security = settings.security
    semantic = load_semantic_detector(
        security.injection_model, security.injection_threshold, security.injection_device
    )
    logger.info(
        "agent.built",
        extra={
            "retrieval_mode": settings.retrieval.mode.value,
            "evidence_reranker": scorer.name if scorer else "term coverage",
            "synthesis": synthesizer.method,
            "verification": f"NLI {nli.name}" if nli else "lexical",
            "injection_action": settings.security.injection_action,
            "query_injection_action": settings.security.query_injection_action,
            "sql_role": settings.tools.sql_user or "application connection",
            "injection_model": semantic.classifier.name if semantic else "rules only",
        },
    )
    return Agent(
        engine=engine,
        registry=build_registry(),
        router=build_router(engine, clock),
        retriever=retriever,
        settings=settings.agent,
        tool_settings=settings.tools,
        synthesizer=synthesizer,
        scorer=scorer,
        term_stats=components.bm25.index,
        clock=clock,
        verification=settings.verification,
        nli=nli,
        security=settings.security,
        policy=load_policy(settings.security.policy_file),
        sql_engine=create_sql_reader_engine(settings) if use_sql_reader else None,
        semantic=semantic,
    )
