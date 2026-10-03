"""The agent: an explicit state machine over ``AgentState``.

    query understanding (+ question screening) -> routing -> tool selection
      -> tool execution (loop) -> evidence aggregation -> security screening
      -> reranking -> evidence validation -> evidence package -> synthesis
      -> claim verification -> output validation -> confidence -> final response

Each stage is a method that updates the state and returns the next stage; the
transitions are written out in ``run``, not hidden in a framework. Every stage is
timed, and any exception inside a stage is recorded as an error rather than
crashing the run. Execution then continues where that is still meaningful:
- a failed tool call is recorded, and the other calls still run;
- a failed synthesis falls back to the extractive answer.
Missing data is reported as a limitation, never filled in.

The agent acts only through the tool registry: every call is validated,
permission-checked and read-only. It has no other access to data or to the system.

Temporal and multi-hop questions (``temporal.py``, ``multihop.py``) are recognised
during query understanding and get structured plans: records selected by timestamps
relative to an anchor, or a chain of recorded links. Disagreeing sources are detected
after reranking (``conflicts.py``) and shown in the answer, never silently dropped.

Security (see ``guard.py``): tools only return rows the caller's grants allow;
security screening then re-checks access, quarantines sources with instructions
aimed at the assistant and redacts credentials, before reranking and before any
model sees the evidence; output validation checks the final answer. Both security
stages fail closed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.engine import Engine

from app.agents.answerability import no_answer_text, suggest_evidence
from app.agents.confidence import compute_confidence
from app.agents.conflicts import find_conflicts, relevant
from app.agents.entities import ServiceCatalog, extract_entities
from app.agents.evidence import EvidenceRanker, EvidenceValidator, TermStatistics, aggregate
from app.agents.guard import WITHHELD, EvidenceScreen, OutputGuard, plain, screen_question
from app.agents.multihop import parse_chain, withheld_notes
from app.agents.planner import ToolPlanner
from app.agents.recommendations import recommend
from app.agents.references import ReferenceFilter
from app.agents.router import RuleBasedRouter
from app.agents.state import (
    AgentError,
    AgentState,
    EvidenceStatus,
    PlannedCall,
    Stage,
    ToolCallRecord,
)
from app.agents.synthesis import CANARY, SYSTEM_PROMPT, ExtractiveSynthesizer, Synthesizer
from app.agents.temporal import TARGET, parse_temporal, timeline_item, tools_for
from app.agents.verification import NLIScorer, Verifier, build_package, citations_for
from app.config import AgentSettings, SecuritySettings, ToolSettings, VerificationSettings
from app.observability import telemetry
from app.rag.reranking.base import PairScorer
from app.rag.retrieval.base import Retriever
from app.schemas.enums import QueryType
from app.security.injection_model import SemanticDetector
from app.security.policy import AccessPolicy, load_policy
from app.security.principal import Principal
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry
from app.tools.trace import TraceChangeOutput

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        engine: Engine,
        registry: ToolRegistry,
        router: RuleBasedRouter,
        retriever: Retriever | None,
        settings: AgentSettings | None = None,
        tool_settings: ToolSettings | None = None,
        synthesizer: Synthesizer | None = None,
        scorer: PairScorer | None = None,
        term_stats: TermStatistics | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        verification: VerificationSettings | None = None,
        nli: NLIScorer | None = None,
        security: SecuritySettings | None = None,
        policy: AccessPolicy | None = None,
        sql_engine: Engine | None = None,
        semantic: SemanticDetector | None = None,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.router = router
        self.retriever = retriever
        self.settings = settings or AgentSettings()
        self.tool_settings = tool_settings or ToolSettings()
        self.synthesizer = synthesizer or ExtractiveSynthesizer(term_stats)
        self.planner = ToolPlanner(self.settings, engine.dialect.name)
        self.ranker = EvidenceRanker(scorer if self.settings.rerank_evidence else None, term_stats)
        self.validator = EvidenceValidator(self.settings.min_query_coverage, term_stats)
        self.verification = verification or VerificationSettings(nli_model=None)
        self.verifier = Verifier(self.verification, nli)
        self.security = security or SecuritySettings()
        self.evidence_screen = EvidenceScreen(
            policy or load_policy(self.security.policy_file),
            self.security.injection_action,
            ReferenceFilter(engine),
            semantic,
        )
        self.output_guard = OutputGuard(SYSTEM_PROMPT, CANARY)
        self.sql_engine = sql_engine  # the read-only role for query_database, if configured
        self.clock = clock

    # --- driver -------------------------------------------------------------------------

    def run(self, query: str, principal: Principal) -> AgentState:
        state = AgentState(
            query=" ".join(query.split())[:1000], principal=principal, now=self.clock()
        )
        stages: dict[Stage, Callable[[AgentState], Stage]] = {
            Stage.UNDERSTAND: self.understand,
            Stage.ROUTE: self.route,
            Stage.SELECT_TOOLS: self.select_tools,
            Stage.EXECUTE: self.execute,
            Stage.AGGREGATE: self.aggregate,
            Stage.SCREEN: self.screen,
            Stage.RERANK: self.rerank,
            Stage.VALIDATE_EVIDENCE: self.validate_evidence,
            Stage.PACKAGE: self.package,
            Stage.SYNTHESIZE: self.synthesize,
            Stage.VERIFY: self.verify,
            Stage.GUARD_OUTPUT: self.guard_output,
            Stage.CONFIDENCE: self.confidence,
            Stage.RESPOND: self.respond,
        }
        fallback = {  # where to continue if a stage raises
            Stage.UNDERSTAND: Stage.ROUTE,
            Stage.ROUTE: Stage.AGGREGATE,
            Stage.SELECT_TOOLS: Stage.AGGREGATE,
            Stage.EXECUTE: Stage.AGGREGATE,
            Stage.AGGREGATE: Stage.SCREEN,
            Stage.SCREEN: Stage.RERANK,  # screen() itself fails closed (drops evidence)
            Stage.RERANK: Stage.VALIDATE_EVIDENCE,
            Stage.VALIDATE_EVIDENCE: Stage.PACKAGE,
            Stage.PACKAGE: Stage.SYNTHESIZE,
            Stage.SYNTHESIZE: Stage.VERIFY,
            Stage.VERIFY: Stage.GUARD_OUTPUT,
            Stage.GUARD_OUTPUT: Stage.CONFIDENCE,  # guard_output() itself fails closed
            Stage.CONFIDENCE: Stage.RESPOND,
            Stage.RESPOND: Stage.DONE,
        }
        started = time.perf_counter()
        stage = Stage.UNDERSTAND
        while stage is not Stage.DONE:
            began = time.perf_counter()
            try:
                next_stage = stages[stage](state)
            except Exception as exc:  # a failing stage must not take the answer down with it
                logger.exception("agent.stage_failed", extra={"stage": stage.value})
                state.errors.append(
                    AgentError(stage=stage, code="stage_failed", message=type(exc).__name__)
                )
                state.limit(
                    f"The {stage.value.replace('_', ' ')} step failed; results may be incomplete."
                )
                next_stage = fallback[stage]
            elapsed = (time.perf_counter() - began) * 1000
            state.latency_ms[stage.value] = round(
                state.latency_ms.get(stage.value, 0.0) + elapsed, 1
            )
            stage = next_stage
        state.latency_ms["total"] = round((time.perf_counter() - started) * 1000, 1)
        telemetry.record_agent(  # tool failures are recorded by the registry itself
            state.query_type.value if state.query_type else None,
            len(state.evidence_package.evidence) if state.evidence_package else 0,
            [f"{e.code}:{e.stage.value}" for e in state.errors if e.tool is None],
            state.latency_ms,
        )
        logger.info(
            "agent.completed",
            extra={
                "user": principal.user_id,
                "query_type": state.query_type.value if state.query_type else None,
                "tool_calls": len(state.tool_results),
                "evidence_status": state.evidence_status.value if state.evidence_status else None,
                "confidence": state.confidence.value,
                "errors": len(state.errors),
                "blocked": state.security.blocked,
                "quarantined": len(state.security.quarantined),
                "total_ms": state.latency_ms["total"],
            },
        )
        return state

    # --- stages ---------------------------------------------------------------------------

    def understand(self, state: AgentState) -> Stage:
        if not screen_question(state, self.security.query_injection_action):
            state.synthesis_method = "none"
            state.step(
                Stage.UNDERSTAND,
                "refused: instruction-like text (" + ", ".join(state.security.query_flags) + ")",
            )
            return Stage.CONFIDENCE
        state.entities = extract_entities(state.query, self.router.services, state.now)
        found = [
            f"{name}={values}"
            for name, values in state.entities.model_dump(exclude={"time_range"}).items()
            if values
        ]
        if state.entities.time_range:
            found.append(f"time={state.entities.time_range.expression}")
        chain = parse_chain(state.query, state.entities)
        temporal = parse_temporal(state.query, state.entities)
        # An explicit relation in time wins over a fix read from a noun ("... before
        # INC-1, excluding hotfixes"); a question about causes or hops stays a chain.
        if chain is not None and not (chain.direction == "fix" and temporal is not None):
            state.chain = chain
        else:
            state.temporal = temporal
        if state.chain:
            found.append(f"chain ({state.chain.direction}) from {state.chain.anchor}")
        if state.temporal:
            t = state.temporal
            found.append(f"temporal: {t.target}s {t.relation.value} {t.anchor_id or 'now'}")
            if t.unhandled:
                state.limit(
                    "The question restricts the answer in a way the timeline does not apply ("
                    + ", ".join(f'"{w}"' for w in t.unhandled)
                    + "); check the listed records against it."
                )
        state.step(Stage.UNDERSTAND, "entities: " + ("; ".join(found) if found else "none"))
        return Stage.ROUTE

    def route(self, state: AgentState) -> Stage:
        decision = self.router.route(
            state.query, permitted=self.registry.permitted(state.principal)
        )
        state.routing, state.query_type = decision, decision.query_type
        state.step(
            Stage.ROUTE,
            f"{decision.query_type.value} ({decision.confidence.value}): {decision.summary}",
        )
        if state.temporal or state.chain:
            # A structured plan answers it; the query type only describes the answer.
            state.query_type = (
                QueryType.MULTI_SOURCE
                if state.chain
                else QueryType.DEPLOYMENT_SEARCH
                if state.temporal and state.temporal.target == "deployment"
                else QueryType.INCIDENT_SEARCH
            )
            state.step(Stage.ROUTE, f"{state.plan} plan; answered as {state.query_type.value}")
            return Stage.SELECT_TOOLS
        for tool in decision.denied_tools:
            state.limit(
                f"{tool} is not permitted for role {state.role!r}; that source was not searched."
            )
        if decision.query_type is QueryType.UNKNOWN:
            return Stage.AGGREGATE
        return Stage.SELECT_TOOLS

    def select_tools(self, state: AgentState) -> Stage:
        state.goals = self.planner.goals(state)
        permitted = self.registry.permitted(state.principal)
        plan = []
        calls = self.planner.initial(state)
        # Every step of a structured plan must be permitted before it starts: a timeline
        # needs the anchor's tool and the target's.
        needed = tools_for(state.temporal) if state.temporal else ["trace_change"]
        if (state.temporal or state.chain) and not all(permitted(t) for t in needed):
            denied = ", ".join(sorted({t for t in needed if not permitted(t)}))
            state.limit(
                f"{denied} is not permitted for role {state.role!r}, so the {state.plan} "
                "plan was not used."
            )
            state.temporal = state.temporal_anchor = state.chain = None
            state.query_type = state.routing.query_type if state.routing else QueryType.UNKNOWN
            if state.query_type is QueryType.UNKNOWN:
                return Stage.AGGREGATE
            state.goals = self.planner.goals(state)
            calls = self.planner.initial(state)
        for call in calls:
            if permitted(call.tool):
                plan.append(call)
            else:
                state.limit(f"{call.tool} is not permitted for role {state.role!r}.")
        if not plan and state.query_type is QueryType.CODE_SEARCH and permitted("search_documents"):
            # A role without code access (SREs, managers) still gets the documentation,
            # which describes configuration keys and components; never the code itself.
            plan = [
                PlannedCall(
                    tool="search_documents",
                    arguments={
                        "query": state.query,
                        "top_k": self.settings.results_per_tool,
                        "services": state.entities.services or None if state.entities else None,
                    },
                    purpose="search the documentation (code search is not permitted)",
                )
            ]
            state.goals = {"documents": False}
            state.limit("The source code was not searched; the answer uses the documentation.")
        state.pending = plan
        state.selected_tools = list(dict.fromkeys(c.tool for c in plan))
        goals = ", ".join(state.goals) or "none"
        planned = ", ".join(c.tool for c in plan) or "no tool"
        state.step(Stage.SELECT_TOOLS, f"plan: {planned}; evidence goals: {goals}")
        return Stage.EXECUTE if plan else Stage.AGGREGATE

    def execute(self, state: AgentState) -> Stage:
        call = state.pending.pop(0)
        done = {c.signature for c, _ in state.outputs} | {
            PlannedCall(tool=r.tool, arguments=r.arguments, purpose="").signature
            for r in state.tool_results
        }
        if call.signature in done:
            return self._next_after_call(state)
        context = ToolContext(
            engine=self.engine,
            principal=state.principal,
            settings=self.tool_settings,
            retriever=self.retriever,
            clock=lambda: state.now,
            sql_engine=self.sql_engine,
        )
        arguments = {k: v for k, v in call.arguments.items() if v is not None}
        result = self.registry.call(call.tool, arguments, context)
        record = ToolCallRecord(
            tool=call.tool,
            purpose=call.purpose,
            arguments=call.arguments,
            status=result.status,
            duration_ms=result.duration_ms,
        )
        if result.ok and result.output is not None:
            record.results = result.output.result_count()
            state.outputs.append((call, result.output))
            self.planner.update_goals(state, call, result.output)
            permitted = self.registry.permitted(state.principal)
            for follow_up in self.planner.follow_ups(state, call, result.output):
                if not permitted(follow_up.tool):
                    state.limit(
                        f"{follow_up.tool} is not permitted for role {state.role!r}; "
                        f"{follow_up.purpose} was skipped."
                    )
                elif follow_up.signature not in done and all(
                    follow_up.signature != p.signature for p in state.pending
                ):
                    state.pending.append(follow_up)
            if follow_up_tools := [
                p.tool for p in state.pending if p.tool not in state.selected_tools
            ]:
                state.selected_tools += list(dict.fromkeys(follow_up_tools))
            state.step(Stage.EXECUTE, f"{call.tool} ({call.purpose}): {record.results} result(s)")
        else:
            message = result.error.message if result.error else "unknown error"
            record.error = message
            state.errors.append(
                AgentError(stage=Stage.EXECUTE, code=result.status, message=message, tool=call.tool)
            )
            state.limit(
                f"{call.tool} failed ({result.status}); its evidence is missing from this answer."
            )
            state.step(Stage.EXECUTE, f"{call.tool} ({call.purpose}): failed, {result.status}")
        state.tool_results.append(record)
        return self._next_after_call(state)

    def _next_after_call(self, state: AgentState) -> Stage:
        if self.planner.satisfied(state):
            if state.pending:
                skipped = ", ".join(p.tool for p in state.pending)
                state.step(Stage.EXECUTE, f"stopped: evidence goals met; skipped {skipped}")
            state.pending = []
            return Stage.AGGREGATE
        if not state.pending:
            return Stage.AGGREGATE
        if len(state.tool_results) >= self.settings.max_tool_calls:
            state.limit(f"Stopped after {self.settings.max_tool_calls} tool calls (the budget).")
            state.pending = []
            return Stage.AGGREGATE
        if (
            sum(r.duration_ms for r in state.tool_results) / 1000
            > self.settings.time_budget_seconds
        ):
            state.limit("Stopped early: the time budget for tool calls ran out.")
            state.pending = []
            return Stage.AGGREGATE
        return Stage.EXECUTE

    def aggregate(self, state: AgentState) -> Stage:
        state.retrieved_documents = aggregate(state.outputs)
        if state.temporal and state.temporal_anchor:
            targets = [o for c, o in state.outputs if c.purpose.startswith(TARGET)]
            if targets:
                state.retrieved_documents.insert(
                    0, timeline_item(state.temporal, state.temporal_anchor, targets[-1])
                )
        for _, output in state.outputs:
            if isinstance(output, TraceChangeOutput):
                for note in withheld_notes(output):
                    state.limit(note)
        kinds: dict[str, int] = {}
        for item in state.retrieved_documents:
            kinds[item.kind.value] = kinds.get(item.kind.value, 0) + 1
        summary = ", ".join(f"{n} {k}" for k, n in kinds.items()) or "nothing"
        state.step(Stage.AGGREGATE, f"{len(state.retrieved_documents)} evidence item(s): {summary}")
        return Stage.SCREEN

    def screen(self, state: AgentState) -> Stage:
        before = len(state.retrieved_documents)
        try:
            state.retrieved_documents = self.evidence_screen.screen(
                state, state.retrieved_documents
            )
        except Exception:  # fail closed: unscreened evidence never goes further
            logger.exception("security.screening_failed", extra={"user": state.user_id})
            state.retrieved_documents = []
            state.evidence_screened = True
            state.errors.append(
                AgentError(
                    stage=Stage.SCREEN,
                    code="screening_failed",
                    message="security screening failed; no evidence was used",
                )
            )
            state.limit("Security screening failed, so no evidence was used.")
        report = state.security
        state.step(
            Stage.SCREEN,
            f"{len(state.retrieved_documents)} of {before} item(s) passed; "
            f"{len(report.quarantined)} quarantined, {len(report.sanitized)} sanitized, "
            f"{report.secrets_redacted} credential(s) and {report.references_redacted} "
            "hidden reference(s) redacted, "
            f"{report.access_violations} access violation(s)",
        )
        return Stage.RERANK

    def rerank(self, state: AgentState) -> Stage:
        state.reranked_evidence = self.ranker.rank(
            state.query, state.retrieved_documents, self.settings.max_evidence
        )
        state.step(
            Stage.RERANK,
            f"kept {len(state.reranked_evidence)} of {len(state.retrieved_documents)} "
            f"by {self.ranker.method}",
        )
        return Stage.VALIDATE_EVIDENCE

    def validate_evidence(self, state: AgentState) -> Stage:
        self.validator.validate(state)
        status = state.evidence_status or EvidenceStatus.INSUFFICIENT
        state.evidence_status = status
        detail = "; ".join(state.evidence_notes[:3])
        state.step(Stage.VALIDATE_EVIDENCE, f"{status.value}" + (f": {detail}" if detail else ""))
        return Stage.PACKAGE

    def package(self, state: AgentState) -> Stage:
        state.evidence_package = build_package(state.query, state.reranked_evidence)
        state.conflicts = find_conflicts(state.reranked_evidence)
        scores = [e.relevance_score for e in state.evidence_package.evidence]
        top = f"; top relevance {max(scores):.2f}" if scores else ""
        state.step(Stage.PACKAGE, f"{len(scores)} item(s) packaged for generation{top}")
        return Stage.SYNTHESIZE

    def synthesize(self, state: AgentState) -> Stage:
        if (
            state.query_type is QueryType.UNKNOWN
            or state.evidence_status is EvidenceStatus.INSUFFICIENT
        ):
            state.suggested_evidence = suggest_evidence(state)
            state.final_answer = no_answer_text(state)
            if state.query_type is not QueryType.UNKNOWN:
                state.final_answer += "\nAdditional evidence that would help:\n" + "\n".join(
                    f"- {idea}" for idea in state.suggested_evidence
                )
                for note in state.evidence_notes:
                    state.limit(note[:1].upper() + note[1:] + ".")
            state.synthesis_method = "none"
        else:
            state.synthesis_method = self.synthesizer.method
            state.final_answer = self.synthesizer.synthesize(state)
            if state.evidence_status is EvidenceStatus.PARTIAL:
                for note in state.evidence_notes:
                    state.limit(note[:1].upper() + note[1:] + ".")
        state.step(Stage.SYNTHESIZE, f"answer written ({state.synthesis_method})")
        return Stage.VERIFY

    def verify(self, state: AgentState) -> Stage:
        if state.synthesis_method == "none" or state.evidence_package is None:
            state.step(Stage.VERIFY, "no claims to verify")
            return Stage.CONFIDENCE
        result = self.verifier.verify(state.final_answer, state.evidence_package)
        state.claims = result.claims
        state.citations = citations_for(result.citations, state.evidence_package)
        unknown = sorted(
            {
                c
                for v in result.claims
                for c in v.cited
                if c not in state.evidence_package.by_label()
            }
        )
        if unknown:
            state.errors.append(
                AgentError(
                    stage=Stage.VERIFY,
                    code="unknown_citation",
                    message="citations to evidence that was never provided were removed: "
                    + ", ".join(unknown),
                )
            )
        counts = result.counts
        removed = [c for c in result.claims if c.action == "removed"]
        if removed:
            state.limit(f"{len(removed)} unsupported claim(s) were removed from the answer.")
        for claim in result.claims:
            if claim.action in {"hedged", "labelled"} and claim.label.value == "UNSUPPORTED":
                state.limit("Some statements could not be verified and are marked as such.")
        if result.dropped_confidence_statements:
            state.limit("Confidence statements written by the language model were ignored.")
        answer = result.text
        if result.notes:
            answer += "\n" + "\n".join(result.notes)
            state.limit("Sources disagree on at least one point; see the notes in the answer.")
            if state.citations:
                # A note cites the disagreeing source ("[E4] state(s) otherwise"): every label
                # in the answer must resolve to a citation (found by the Phase 14 audit).
                disagreeing = [
                    label
                    for claim in result.claims
                    if claim.action != "removed"
                    for label in claim.conflicting
                ]
                state.citations = citations_for(
                    [c.label for c in state.citations] + disagreeing, state.evidence_package
                )
        services = state.entities.services if state.entities else []
        state.conflicts = [c for c in state.conflicts if relevant(c, state.query, answer, services)]
        if state.conflicts and state.citations:
            # Every disagreeing source is shown with its date; none is dropped.
            answer += "\n" + "\n".join(c.describe() for c in state.conflicts)
            labels = [label for c in state.conflicts for v in c.values for label in v.labels]
            state.citations = citations_for(
                [c.label for c in state.citations] + labels, state.evidence_package
            )
            state.limit(
                "Sources disagree on "
                + ", ".join(c.setting for c in state.conflicts)
                + "; every value is shown with its source and date."
            )
        if not state.citations:  # nothing verifiable survived
            state.suggested_evidence = suggest_evidence(state)
            answer = (
                no_answer_text(state)
                + "\nAdditional evidence that would help:\n"
                + "\n".join(f"- {idea}" for idea in state.suggested_evidence)
            )
        state.final_answer = answer
        state.step(
            Stage.VERIFY,
            f"{len(result.claims)} claim(s) by {self.verifier.method}: "
            f"{counts['SUPPORTED']} supported, {counts['PARTIALLY_SUPPORTED']} partial, "
            f"{counts['UNSUPPORTED']} unsupported; {len(state.citations)} source(s) cited",
        )
        return Stage.GUARD_OUTPUT

    def guard_output(self, state: AgentState) -> Stage:
        before = state.final_answer
        failed = False
        try:
            self.output_guard.check(state)
        except Exception:  # fail closed: an answer that was not validated is withheld
            logger.exception("security.output_validation_failed", extra={"user": state.user_id})
            failed = True
            state.final_answer = WITHHELD
            state.limit("The answer was withheld: it could not be validated.")
            state.errors.append(
                AgentError(
                    stage=Stage.GUARD_OUTPUT,
                    code="output_validation_failed",
                    message="the answer could not be validated and was withheld",
                )
            )
        if state.final_answer != before:
            # Keep only the citations and claims still in the answer.
            state.citations = [c for c in state.citations if f"[{c.label}]" in state.final_answer]
            kept_text = plain(state.final_answer)
            state.claims = [
                c
                if c.action == "removed" or plain(c.text) in kept_text
                else c.model_copy(update={"action": "removed", "reason": "output validation"})
                for c in state.claims
            ]
            if not state.citations and state.synthesis_method != "none" and not failed:
                state.final_answer = no_answer_text(state)
        removed = state.security.output_removed
        state.step(
            Stage.GUARD_OUTPUT,
            "answer passed" if not removed else "removed: " + ", ".join(removed),
        )
        return Stage.CONFIDENCE

    def confidence(self, state: AgentState) -> Stage:
        breakdown = compute_confidence(state, self.verification)
        state.confidence_breakdown = breakdown
        state.confidence = breakdown.level
        state.confidence_reasons = breakdown.reasons
        state.step(Stage.CONFIDENCE, f"{breakdown.level.value} (score {breakdown.overall:.2f})")
        return Stage.RESPOND

    def respond(self, state: AgentState) -> Stage:
        state.recommendations = recommend(state)
        state.step(
            Stage.RESPOND,
            f"{state.confidence.value} confidence; {len(state.tool_results)} tool call(s), "
            f"{len(state.errors)} error(s)",
        )
        return Stage.DONE


def build_router(engine: Engine, clock: Callable[[], datetime]) -> RuleBasedRouter:
    return RuleBasedRouter(ServiceCatalog.from_engine(engine), clock=clock)
