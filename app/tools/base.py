"""The tool contract.

A tool is a fixed, read-only capability with a typed input model, a typed output
model and one required permission. ``Tool.execute`` is the only entry point:

    raw arguments -> input model (validated, extra fields rejected)
                  -> permission check (the caller's role must hold ``permission``)
                  -> ``_run`` (applies the caller's clearance to every query)
                  -> output model (type-checked)

The registry calls ``execute``, and so does any direct caller: a tool cannot run
without validation and the permission check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.engine import Engine

from app.config import ToolSettings
from app.rag.retrieval.base import RetrievedChunk, Retriever
from app.schemas.enums import AccessLevel, SourceType, ToolPermission
from app.security.principal import Principal

# --- errors -----------------------------------------------------------------------


class ToolError(Exception):
    """Base class; ``code`` is the machine-readable status reported to the agent."""

    code: ClassVar[str] = "error"

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or []


class ToolNotFoundError(ToolError):
    code = "unknown_tool"


class ToolInputError(ToolError):
    code = "invalid_input"


class UnsafeSqlError(ToolInputError):
    code = "unsafe_sql"


class ToolPermissionError(ToolError):
    code = "permission_denied"


class ResourceNotFoundError(ToolError):
    """Nothing matched, or it exists but is above the caller's clearance (the two are
    deliberately indistinguishable)."""

    code = "not_found"


class ToolExecutionError(ToolError):
    code = "execution_error"


# --- context and shared models -------------------------------------------------------


@dataclass
class ToolContext:
    """Everything a tool may use. Tools get no other handles (no filesystem, no shell)."""

    engine: Engine
    principal: Principal
    settings: ToolSettings = field(default_factory=ToolSettings)
    retriever: Retriever | None = None  # text search (dense / BM25 / hybrid / reranked)
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    # query_database runs through this engine when set: a read-only database role.
    sql_engine: Engine | None = None

    def require_retriever(self) -> Retriever:
        if self.retriever is None:
            raise ToolExecutionError("no retriever is configured for text search")
        return self.retriever


# Output fields that hold a tool's records (the first one present is counted).
RESULT_FIELDS = ("results", "incidents", "deployments", "entries", "rows")


class ToolModel(BaseModel):
    """Base for tool inputs and outputs: unknown fields are rejected, values frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def result_count(self) -> int:
        """How many records this output holds: the length of its result list, else one.
        Only the model's own declared fields are read (no dynamic attribute access)."""
        declared = type(self).model_fields
        for name in RESULT_FIELDS:
            if name in declared and isinstance(self.__dict__[name], list):
                return len(self.__dict__[name])
        return 1


class Evidence(ToolModel):
    """A retrieved chunk with its provenance: enough to cite and verify it."""

    chunk_id: str
    document_id: str
    source_type: SourceType
    title: str
    section: str | None
    service_id: str | None
    timestamp: datetime
    access_level: AccessLevel
    version: str | None
    file_path: str | None
    content: str
    content_truncated: bool
    score: float
    rank: int
    retriever: str
    score_details: dict[str, float] = {}
    matched_terms: tuple[str, ...] = ()

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, max_chars: int) -> Evidence:
        truncated = len(chunk.content) > max_chars
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_type=chunk.source_type,
            title=chunk.title,
            section=chunk.section,
            service_id=chunk.service_id,
            timestamp=chunk.timestamp,
            access_level=chunk.access_level,
            version=chunk.version,
            file_path=chunk.file_path,
            content=chunk.content[:max_chars],
            content_truncated=truncated,
            score=chunk.score,
            rank=chunk.rank,
            retriever=chunk.retriever,
            score_details=chunk.score_details,
            matched_terms=chunk.matched_terms,
        )


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


class ToolSpec(ToolModel):
    """What the future agent is shown about a tool."""

    name: str
    description: str
    permission: ToolPermission
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


# --- the tool -----------------------------------------------------------------------

InputT = TypeVar("InputT", bound=ToolModel)
OutputT = TypeVar("OutputT", bound=ToolModel)


class Tool(ABC, Generic[InputT, OutputT]):
    name: ClassVar[str]
    description: ClassVar[str]
    permission: ClassVar[ToolPermission]
    input_model: ClassVar[type[ToolModel]]
    output_model: ClassVar[type[ToolModel]]

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            permission=self.permission,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
        )

    def parse(self, arguments: Mapping[str, Any] | ToolModel) -> InputT:
        if isinstance(arguments, self.input_model):
            return arguments  # type: ignore[return-value]
        if isinstance(arguments, BaseModel):
            raise ToolInputError(
                f"{self.name} expects {self.input_model.__name__}, got {type(arguments).__name__}"
            )
        if not isinstance(arguments, Mapping):
            raise ToolInputError(f"{self.name} arguments must be an object")
        try:
            return self.input_model.model_validate(dict(arguments))  # type: ignore[return-value]
        except ValidationError as exc:
            details = [
                {"field": ".".join(str(p) for p in e["loc"]), "error": e["msg"]}
                for e in exc.errors(include_url=False)
            ]
            raise ToolInputError(f"invalid arguments for {self.name}", details) from exc

    def execute(self, arguments: Mapping[str, Any] | ToolModel, context: ToolContext) -> OutputT:
        """Validate, check the permission, run, and type-check the output."""
        if not context.principal.can(self.permission):
            raise ToolPermissionError(
                f"role {context.principal.role!r} may not call {self.name} "
                f"(requires {self.permission.value})"
            )
        parsed = self.parse(arguments)
        output = self._run(parsed, context)
        if not isinstance(output, self.output_model):
            raise ToolExecutionError(f"{self.name} returned {type(output).__name__}")
        return output

    @abstractmethod
    def _run(self, arguments: InputT, context: ToolContext) -> OutputT:
        """The tool's work. Must apply ``context.principal`` clearance to all data."""
