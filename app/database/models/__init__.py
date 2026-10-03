"""ORM models. Importing this package registers every table on ``Base.metadata``."""

from app.database.models.auth import ApiToken
from app.database.models.delivery import CodeFile, Deployment, PullRequest, PullRequestFile
from app.database.models.identity import Role, User
from app.database.models.knowledge import ChunkEmbedding, Document, DocumentChunk
from app.database.models.operations import Incident, LogEntry
from app.database.models.services import Service, ServiceDependency

__all__ = [
    "ApiToken",
    "ChunkEmbedding",
    "CodeFile",
    "Deployment",
    "Document",
    "DocumentChunk",
    "Incident",
    "LogEntry",
    "PullRequest",
    "PullRequestFile",
    "Role",
    "Service",
    "ServiceDependency",
    "User",
]
