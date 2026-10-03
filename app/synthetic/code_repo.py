"""Renders the NovaCart monorepo at HEAD from the service catalog.

Generic files (config, routes, database, cache, events, clients, tests, manifests)
come from templates parameterised by each ``ServiceProfile``; hand-written domain
modules come from ``domain_code``. Change kinds (``changes.py``) diff against
exactly these contents, so patches and files always agree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.enums import AccessLevel, CodeFileKind, ServiceTier
from app.synthetic.catalog import (
    INTERNAL_DOMAIN,
    KAFKA_BOOTSTRAP,
    SERVICES,
    SERVICES_BY_ID,
    TOPIC_PARTITIONS,
    TOPIC_PRODUCERS,
    ExternalProvider,
    ServiceProfile,
)
from app.synthetic.domain_code import domain_module_source, primary_class
from app.synthetic.text import bullet_list, markdown_table, render


@dataclass
class RepoFile:
    path: str
    service_id: str | None
    language: str
    kind: CodeFileKind
    content: str
    access_level: AccessLevel = AccessLevel.ENGINEERING


# --- Database tables ----------------------------------------------------------------

# table -> (ORM class, [(column, SQLAlchemy type)])
TABLE_MODELS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "credentials": (
        "Credential",
        [
            ("user_id", "String(36)"),
            ("email", "String(256)"),
            ("password_hash", "String(256)"),
            ("status", "String(32)"),
            ("failed_attempts", "Integer"),
        ],
    ),
    "refresh_tokens": (
        "RefreshToken",
        [
            ("user_id", "String(36)"),
            ("token_hash", "String(64)"),
            ("expires_at", "DateTime(timezone=True)"),
            ("revoked", "Boolean"),
        ],
    ),
    "signing_keys": (
        "SigningKey",
        [
            ("kid", "String(64)"),
            ("algorithm", "String(16)"),
            ("public_jwk", "Text"),
            ("status", "String(32)"),
            ("activated_at", "DateTime(timezone=True)"),
        ],
    ),
    "users": (
        "User",
        [
            ("email", "String(256)"),
            ("display_name", "String(128)"),
            ("status", "String(32)"),
            ("locale", "String(8)"),
        ],
    ),
    "addresses": (
        "Address",
        [
            ("user_id", "String(36)"),
            ("line1", "String(256)"),
            ("city", "String(128)"),
            ("postal_code", "String(16)"),
            ("country", "String(2)"),
            ("is_default", "Boolean"),
        ],
    ),
    "preferences": (
        "Preference",
        [("user_id", "String(36)"), ("key", "String(64)"), ("value", "String(256)")],
    ),
    "products": (
        "Product",
        [
            ("title", "String(256)"),
            ("brand", "String(128)"),
            ("category", "String(128)"),
            ("status", "String(32)"),
        ],
    ),
    "skus": (
        "Sku",
        [
            ("product_id", "String(36)"),
            ("sku", "String(64)"),
            ("color", "String(32)"),
            ("size", "String(16)"),
        ],
    ),
    "prices": (
        "Price",
        [
            ("sku", "String(64)"),
            ("list_price_minor", "Integer"),
            ("currency", "String(3)"),
            ("valid_from", "DateTime(timezone=True)"),
        ],
    ),
    "stock_levels": (
        "StockLevel",
        [
            ("sku", "String(64)"),
            ("warehouse_id", "String(36)"),
            ("on_hand", "Integer"),
            ("reserved", "Integer"),
        ],
    ),
    "reservations": (
        "Reservation",
        [
            ("order_id", "String(36)"),
            ("sku", "String(64)"),
            ("quantity", "Integer"),
            ("status", "String(32)"),
            ("expires_at", "DateTime(timezone=True)"),
        ],
    ),
    "warehouses": (
        "Warehouse",
        [("code", "String(16)"), ("region", "String(32)"), ("status", "String(32)")],
    ),
    "orders": (
        "Order",
        [
            ("user_id", "String(36)"),
            ("cart_id", "String(36)"),
            ("total_minor", "Integer"),
            ("currency", "String(3)"),
            ("status", "String(32)"),
            ("payment_id", "String(36)"),
        ],
    ),
    "order_items": (
        "OrderItem",
        [
            ("order_id", "String(36)"),
            ("sku", "String(64)"),
            ("quantity", "Integer"),
            ("unit_price_minor", "Integer"),
        ],
    ),
    "order_events": (
        "OrderEvent",
        [("order_id", "String(36)"), ("event_type", "String(64)"), ("payload", "Text")],
    ),
    "payments": (
        "Payment",
        [
            ("order_id", "String(36)"),
            ("amount_minor", "Integer"),
            ("currency", "String(3)"),
            ("status", "String(32)"),
            ("psp_reference", "String(64)"),
            ("idempotency_key", "String(128)"),
        ],
    ),
    "refunds": (
        "Refund",
        [
            ("payment_id", "String(36)"),
            ("amount_minor", "Integer"),
            ("reason", "String(256)"),
            ("status", "String(32)"),
        ],
    ),
    "idempotency_keys": (
        "IdempotencyKey",
        [
            ("key", "String(128)"),
            ("response_hash", "String(64)"),
            ("expires_at", "DateTime(timezone=True)"),
        ],
    ),
    "ledger_entries": (
        "LedgerEntry",
        [
            ("payment_id", "String(36)"),
            ("account", "String(64)"),
            ("direction", "String(6)"),
            ("amount_minor", "Integer"),
        ],
    ),
    "notifications": (
        "Notification",
        [
            ("user_id", "String(36)"),
            ("channel", "String(16)"),
            ("template", "String(64)"),
            ("status", "String(32)"),
        ],
    ),
    "delivery_attempts": (
        "DeliveryAttempt",
        [
            ("notification_id", "String(36)"),
            ("provider", "String(32)"),
            ("response_code", "Integer"),
            ("attempted_at", "DateTime(timezone=True)"),
        ],
    ),
    "templates": (
        "Template",
        [
            ("name", "String(64)"),
            ("channel", "String(16)"),
            ("body", "Text"),
            ("version", "Integer"),
        ],
    ),
}

_PY_TYPES = {"Integer": "int", "Boolean": "bool", "Text": "str"}


def _py_type(sa_type: str) -> str:
    if sa_type.startswith("DateTime"):
        return "datetime"
    return _PY_TYPES.get(sa_type, "str")


def primary_table(profile: ServiceProfile) -> str:
    """First table with a ``status`` column: the aggregate the repository manages."""
    for table in profile.tables:
        if any(column == "status" for column, _ in TABLE_MODELS[table][1]):
            return table
    raise ValueError(f"{profile.id} has no table with a status column")


def repository_class(profile: ServiceProfile) -> str:
    return TABLE_MODELS[primary_table(profile)][0] + "Repository"


# --- Kafka events -------------------------------------------------------------------

EVENT_FIELDS: dict[str, list[str]] = {
    "user.logged_in": ["user_id", "method", "ip_country"],
    "user.updated": ["user_id", "changed_fields"],
    "product.updated": ["product_id", "sku_ids", "status"],
    "price.changed": ["sku", "old_price_minor", "new_price_minor", "currency"],
    "inventory.reserved": ["reservation_id", "order_id", "sku_count"],
    "inventory.depleted": ["sku", "warehouse_id"],
    "order.created": ["order_id", "user_id", "total_minor", "currency"],
    "order.cancelled": ["order_id", "reason"],
    "payment.completed": ["payment_id", "order_id", "amount_minor", "currency"],
    "payment.failed": ["payment_id", "order_id", "decline_code"],
    "refund.issued": ["refund_id", "payment_id", "amount_minor"],
}


def event_class(topic: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[._]", topic))


def _event_field_type(name: str) -> str:
    if name.endswith(("_minor", "_count")):
        return "int"
    if name in {"changed_fields", "sku_ids"}:
        return "list[str]"
    return "str"


def handler_name(topic: str) -> str:
    return "on_" + topic.replace(".", "_")


# --- HTTP clients -------------------------------------------------------------------

# Target service -> (method, HTTP verb, path, params) used by callers' clients.
CLIENT_METHODS: dict[str, tuple[str, str, str, list[str]]] = {
    "auth-service": ("fetch_jwks", "get", "/v1/auth/.well-known/jwks.json", []),
    "user-service": ("get_user", "get", "/v1/users/{user_id}", ["user_id"]),
    "product-service": ("get_product", "get", "/v1/products/{product_id}", ["product_id"]),
    "inventory-service": ("reserve", "post", "/v1/reservations", ["order_id", "lines"]),
    "cart-service": ("get_cart", "get", "/v1/carts/{cart_id}", ["cart_id"]),
    "payment-service": (
        "authorize",
        "post",
        "/v1/payments",
        ["order_id", "amount", "currency", "idempotency_key"],
    ),
    "search-service": ("search", "get", "/v1/search", ["q"]),
    "recommendation-service": ("recommend", "get", "/v1/recommendations/{user_id}", ["user_id"]),
    "order-service": ("get_order", "get", "/v1/orders/{order_id}", ["order_id"]),
}


def client_module(service_id: str) -> str:
    return SERVICES_BY_ID[service_id].short.lower() + "_client"


def client_class(service_id: str) -> str:
    return SERVICES_BY_ID[service_id].short + "Client"


def dependency_url_setting(service_id: str) -> str:
    return SERVICES_BY_ID[service_id].package + "_url"


def service_url(service_id: str) -> str:
    target = SERVICES_BY_ID[service_id]
    return f"http://{target.id}.novacart.svc.cluster.local:{target.port}"


def has_http_clients(profile: ServiceProfile) -> bool:
    # The gateway proxies generically (routing.py) instead of using typed clients.
    return profile.id != "api-gateway" and bool(profile.http_dependencies or profile.external)


# --- Generic service files ------------------------------------------------------------


def _config_py(p: ServiceProfile) -> str:
    c = p.config
    lines = [
        f'"""Runtime configuration for the {p.display_name}.',
        "",
        f"Every field can be overridden by an environment variable prefixed ``{p.env_prefix}_``",
        f"(for example ``{p.env_prefix}_HTTP_TIMEOUT_SECONDS``). Production values are set in",
        f"``deploy/{p.id}.yaml``.",
        '"""',
        "",
        "from pydantic_settings import BaseSettings, SettingsConfigDict",
        "",
        "",
        "class Settings(BaseSettings):",
        f'    model_config = SettingsConfigDict(env_prefix="{p.env_prefix}_")',
        "",
        f'    service_name: str = "{p.id}"',
        f"    port: int = {p.port}",
        '    log_level: str = "INFO"',
        f"    http_timeout_seconds: float = {c.http_timeout_seconds}",
        f"    http_max_retries: int = {c.http_max_retries}",
    ]
    if p.postgres_db:
        lines += [
            f'    database_url: str = "{database_url(p)}"',
            f"    db_pool_size: int = {c.db_pool_size}",
            f"    db_max_overflow: int = {c.db_max_overflow}",
            f"    db_pool_timeout_seconds: float = {c.db_pool_timeout_seconds}",
        ]
    if p.redis_cluster:
        lines += [
            f'    redis_url: str = "{redis_url(p)}"',
            f"    cache_ttl_seconds: int = {c.cache_ttl_seconds}",
        ]
    if p.produces or p.consumes:
        lines.append(f'    kafka_bootstrap_servers: str = "{KAFKA_BOOTSTRAP}"')
    if p.consumes:
        lines += [
            f'    kafka_consumer_group: str = "{p.consumer_group}"',
            f"    kafka_max_poll_records: int = {c.kafka_max_poll_records}",
        ]
    if "opensearch:search-products" in p.extra_datastores:
        lines.append(f'    opensearch_url: str = "https://search-products.{INTERNAL_DOMAIN}:9200"')
    if p.id != "api-gateway":
        for dep in p.http_dependencies:
            lines.append(
                f'    {dependency_url_setting(dep.service)}: str = "{service_url(dep.service)}"'
            )
    for provider in p.external:
        lines += [
            f'    {provider.settings_prefix}_base_url: str = "{provider.base_url}"',
            f'    {provider.settings_prefix}_api_key: str = ""  # injected from Vault at runtime',
        ]
    lines += ["", "", "settings = Settings()", ""]
    return "\n".join(lines)


def database_url(p: ServiceProfile) -> str:
    return f"postgresql+psycopg://{p.postgres_db}_app@pgbouncer-{p.postgres_db}.{INTERNAL_DOMAIN}:6432/{p.postgres_db}"


def redis_url(p: ServiceProfile) -> str:
    return f"redis://{p.redis_cluster}.{INTERNAL_DOMAIN}:6379/0"


def _deploy_yaml(p: ServiceProfile) -> str:
    c = p.config
    strategy = "canary" if p.tier is ServiceTier.TIER_0 else "rolling"
    env = [
        (f"{p.env_prefix}_LOG_LEVEL", "INFO"),
        (f"{p.env_prefix}_HTTP_TIMEOUT_SECONDS", str(c.http_timeout_seconds)),
        (f"{p.env_prefix}_HTTP_MAX_RETRIES", str(c.http_max_retries)),
    ]
    if p.postgres_db:
        env += [
            (f"{p.env_prefix}_DATABASE_URL", database_url(p)),
            (f"{p.env_prefix}_DB_POOL_SIZE", str(c.db_pool_size)),
            (f"{p.env_prefix}_DB_MAX_OVERFLOW", str(c.db_max_overflow)),
        ]
    if p.redis_cluster:
        env += [
            (f"{p.env_prefix}_REDIS_URL", redis_url(p)),
            (f"{p.env_prefix}_CACHE_TTL_SECONDS", str(c.cache_ttl_seconds)),
        ]
    if p.consumes:
        env.append((f"{p.env_prefix}_KAFKA_MAX_POLL_RECORDS", str(c.kafka_max_poll_records)))
    if p.id != "api-gateway":
        for dep in p.http_dependencies:
            env.append(
                (
                    f"{p.env_prefix}_{dependency_url_setting(dep.service).upper()}",
                    service_url(dep.service),
                )
            )
    env_lines = "\n".join(f'  {key}: "{value}"' for key, value in env)
    secrets = [f"{p.env_prefix}_{prov.settings_prefix.upper()}_API_KEY" for prov in p.external]
    if p.postgres_db:
        secrets.append(f"{p.env_prefix}_DATABASE_PASSWORD")
    secret_lines = (
        "\n".join(
            f"  - name: {name}\n    vault_path: secret/{p.id}/{name.lower()}" for name in secrets
        )
        or "  []"
    )
    return render(
        """
        # Production manifest for {{id}} (rendered by the platform Helm chart).
        service: {{id}}
        team: {{team}}
        tier: {{tier}}
        image: registry.novacart.internal/{{id}}
        replicas: {{replicas}}
        strategy: {{strategy}}
        resources:
          requests:
            cpu: {{cpu}}
            memory: {{memory}}
          limits:
            memory: {{memory}}
        probes:
          readiness: /healthz
          liveness: /healthz
        autoscaling:
          min_replicas: {{replicas}}
          max_replicas: {{max_replicas}}
          target_cpu_utilization: 65
        env:
        {{env}}
        secrets:
        {{secrets}}
        """,
        id=p.id,
        team=p.team,
        tier=p.tier.value,
        replicas=c.replicas,
        strategy=strategy,
        cpu=c.cpu,
        memory=c.memory,
        max_replicas=c.replicas * 3,
        env=env_lines,
        secrets=secret_lines,
    )


def _route_params(method: str, path: str, handler: str) -> tuple[list[str], list[str]]:
    params = [(name, "str") for name in re.findall(r"\{(\w+)\}", path)]
    args = [name for name, _ in params]
    signature = [f"{name}: {kind}" for name, kind in params]
    if handler in {"search_products", "suggest"}:
        signature += ["q: str", "limit: int = 20"]
        args += ["q", "limit"]
    elif handler.startswith("list_"):
        signature.append("limit: int = 50")
        args.append("limit")
    if method in {"POST", "PUT", "PATCH"}:
        signature.append("payload: dict[str, Any] | None = None")
        args.append("payload")
    return signature, args


def _routes_py(p: ServiceProfile) -> str:
    module, cls = primary_class(p.id)
    blocks = []
    for ep in p.endpoints:
        signature, args = _route_params(ep.method, ep.path, ep.handler)
        # "handler", not "service": the gateway has a path parameter named `service`.
        signature.append(f"handler: {cls} = Depends(get_service)")
        call_args = ", ".join(f"{name}={name}" for name in args)
        blocks.append(
            f'@router.{ep.method.lower()}("{ep.path}")\n'
            f"async def {ep.handler}(\n    "
            + ",\n    ".join(signature)
            + ",\n) -> dict[str, Any]:\n"
            f'    """{ep.summary}."""\n'
            f"    return await handler.{ep.handler}({call_args})\n"
        )
    return (
        f'"""HTTP routes for the {p.display_name}."""\n\n'
        "from __future__ import annotations\n\n"
        "from functools import lru_cache\n"
        "from typing import Any\n\n"
        "from fastapi import APIRouter, Depends\n\n"
        f"from {p.package}.{module} import {cls}\n\n"
        "router = APIRouter()\n\n\n"
        "@lru_cache(maxsize=1)\n"
        f"def get_service() -> {cls}:\n"
        f"    return {cls}.from_settings()\n\n\n" + "\n\n".join(blocks)
    )


def _main_py(p: ServiceProfile) -> str:
    consumer_import = (
        f"from {p.package}.events.consumer import EventConsumer\n" if p.consumes else ""
    )
    consumer_start = (
        "    consumer = EventConsumer()\n"
        "    consumer_task = asyncio.create_task(consumer.start())\n"
        if p.consumes
        else ""
    )
    consumer_stop = "    consumer.stop()\n    consumer_task.cancel()\n" if p.consumes else ""
    asyncio_import = "import asyncio\n" if p.consumes else ""
    return (
        f'"""{p.display_name} application entry point."""\n\n'
        "from __future__ import annotations\n\n"
        f"{asyncio_import}"
        "from collections.abc import AsyncIterator\n"
        "from contextlib import asynccontextmanager\n\n"
        "from fastapi import FastAPI, Request\n\n"
        "from novacart_common.logging import configure_logging, get_logger\n"
        "from novacart_common.tracing import new_request_id\n"
        f"from {p.package}.api.routes import router\n"
        f"from {p.package}.config import settings\n"
        f"{consumer_import}\n"
        "logger = get_logger(__name__)\n\n\n"
        "@asynccontextmanager\n"
        "async def lifespan(app: FastAPI) -> AsyncIterator[None]:\n"
        "    configure_logging(settings.service_name, settings.log_level)\n"
        f"{consumer_start}"
        '    logger.info("service.started", extra={"port": settings.port})\n'
        "    yield\n"
        f"{consumer_stop}"
        '    logger.info("service.stopped")\n\n\n'
        f'app = FastAPI(title="{p.display_name}", lifespan=lifespan)\n'
        "app.include_router(router)\n\n\n"
        '@app.middleware("http")\n'
        "async def add_request_context(request: Request, call_next):\n"
        '    request.state.request_id = request.headers.get("x-request-id") or new_request_id()\n'
        "    response = await call_next(request)\n"
        '    response.headers["x-request-id"] = request.state.request_id\n'
        "    return response\n\n\n"
        '@app.get("/healthz")\n'
        "async def healthz() -> dict[str, str]:\n"
        '    return {"status": "ok", "service": settings.service_name}\n'
    )


def _database_py(p: ServiceProfile) -> str:
    return render(
        '''
        """Database engine and session management for the {{display}}.

        Connections go through PgBouncer (transaction pooling). Pool sizing is per
        pod: total server connections = replicas x (db_pool_size + db_max_overflow).
        """

        from __future__ import annotations

        from collections.abc import Iterator
        from contextlib import contextmanager

        from sqlalchemy import create_engine, event
        from sqlalchemy.engine import Engine
        from sqlalchemy.orm import Session, sessionmaker

        from novacart_common.logging import get_logger
        from {{pkg}}.config import settings

        logger = get_logger(__name__)

        engine: Engine = create_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout_seconds,
            pool_recycle=1800,
            pool_pre_ping=True,
        )

        SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


        @event.listens_for(engine, "checkout")
        def _on_checkout(dbapi_connection, connection_record, connection_proxy) -> None:
            pool = engine.pool
            if pool.checkedout() >= pool.size():
                logger.warning(
                    "db.pool.high_watermark",
                    extra={"checked_out": pool.checkedout(), "size": pool.size(), "overflow": pool.overflow()},
                )


        @contextmanager
        def session_scope() -> Iterator[Session]:
            """Transactional scope: commit on success, roll back on error, always close."""
            session = SessionLocal()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()


        def pool_status() -> dict[str, int]:
            pool = engine.pool
            return {"size": pool.size(), "checked_out": pool.checkedout(), "overflow": pool.overflow()}
        ''',
        display=p.display_name,
        pkg=p.package,
    )


def _models_py(p: ServiceProfile) -> str:
    classes = []
    used = {"String", "DateTime", "func"}
    for table in p.tables:
        cls, columns = TABLE_MODELS[table]
        lines = [
            f"class {cls}(Base):",
            f'    __tablename__ = "{table}"',
            "",
            "    id: Mapped[str] = mapped_column(String(36), primary_key=True)",
        ]
        for name, sa_type in columns:
            used.add(re.match(r"\w+", sa_type).group(0))  # type: ignore[union-attr]
            lines.append(
                f"    {name}: Mapped[{_py_type(sa_type)}] = mapped_column({sa_type}, nullable=False)"
            )
        lines += [
            "    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())",
            "    updated_at: Mapped[datetime] = mapped_column(",
            "        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()",
            "    )",
            "",
            "    def to_dict(self) -> dict[str, object]:",
            "        return {column.name: getattr(self, column.name) for column in self.__table__.columns}",
        ]
        classes.append("\n".join(lines))
    imports = ", ".join([*sorted(used - {"func"}), "func"])
    return (
        f'"""SQLAlchemy models for the ``{p.postgres_db}`` database."""\n\n'
        "from __future__ import annotations\n\n"
        "from datetime import datetime\n\n"
        f"from sqlalchemy import {imports}\n"
        "from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column\n\n\n"
        "class Base(DeclarativeBase):\n    pass\n\n\n" + "\n\n\n".join(classes) + "\n"
    )


def _repository_py(p: ServiceProfile) -> str:
    model = TABLE_MODELS[primary_table(p)][0]
    return render(
        '''
        """Data access for the {{display}} ({{table}} table)."""

        from __future__ import annotations

        from typing import Any

        from sqlalchemy import select, update

        from novacart_common.ids import new_id
        from novacart_common.logging import get_logger
        from {{pkg}}.db.database import session_scope
        from {{pkg}}.db.models import {{model}}

        logger = get_logger(__name__)


        class {{model}}Repository:
            def get(self, record_id: str) -> dict[str, Any] | None:
                with session_scope() as session:
                    row = session.get({{model}}, record_id)
                    return row.to_dict() if row else None

            def create(self, **values: Any) -> dict[str, Any]:
                with session_scope() as session:
                    row = {{model}}(id=new_id(), **values)
                    session.add(row)
                    session.flush()
                    return row.to_dict()

            def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
                with session_scope() as session:
                    rows = session.scalars(
                        select({{model}}).order_by({{model}}.created_at.desc()).limit(limit)
                    )
                    return [row.to_dict() for row in rows]

            def bulk_update_status(self, record_ids: list[str], status: str) -> int:
                """Update many rows in one transaction.

                Rows are locked in primary-key order so that concurrent callers acquire
                row locks in the same order and cannot deadlock each other.
                """
                with session_scope() as session:
                    updated = 0
                    for record_id in sorted(record_ids):
                        session.execute(
                            select({{model}}).where({{model}}.id == record_id).with_for_update()
                        )
                        result = session.execute(
                            update({{model}}).where({{model}}.id == record_id).values(status=status)
                        )
                        updated += result.rowcount
                    logger.info("repository.bulk_update", extra={"rows": updated, "status": status})
                    return updated
        ''',
        display=p.display_name,
        table=primary_table(p),
        pkg=p.package,
        model=model,
    )


def _migration_initial(p: ServiceProfile) -> str:
    tables = []
    for table in p.tables:
        _, columns = TABLE_MODELS[table]
        cols = ['        sa.Column("id", sa.String(36), primary_key=True),']
        cols += [
            f'        sa.Column("{name}", sa.{sa_type}, nullable=False),'
            for name, sa_type in columns
        ]
        cols += [
            '        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),',
            '        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),',
        ]
        tables.append(f'    op.create_table(\n        "{table}",\n' + "\n".join(cols) + "\n    )")
    drops = "\n".join(f'    op.drop_table("{table}")' for table in reversed(p.tables))
    return (
        f'"""Initial schema for the {p.postgres_db} database.\n\nRevision ID: 0001\nRevises:\n"""\n\n'
        "import sqlalchemy as sa\nfrom alembic import op\n\n"
        'revision = "0001"\ndown_revision = None\n\n\n'
        "def upgrade() -> None:\n" + "\n".join(tables) + "\n\n\n"
        "def downgrade() -> None:\n" + drops + "\n"
    )


def _migration_indexes(p: ServiceProfile) -> str:
    creates, drops = [], []
    for table in p.tables:
        names = [name for name, _ in TABLE_MODELS[table][1]]
        if "status" in names:
            index = f"ix_{table}_status_created_at"
            creates.append(f'    op.create_index("{index}", "{table}", ["status", "created_at"])')
            drops.append(f'    op.drop_index("{index}", table_name="{table}")')
        for fk in ("order_id", "payment_id", "user_id", "sku", "notification_id"):
            if fk in names:
                index = f"ix_{table}_{fk}"
                creates.append(f'    op.create_index("{index}", "{table}", ["{fk}"])')
                drops.append(f'    op.drop_index("{index}", table_name="{table}")')
    return (
        '"""Add lookup indexes.\n\nRevision ID: 0002\nRevises: 0001\n"""\n\n'
        "from alembic import op\n\n"
        'revision = "0002"\ndown_revision = "0001"\n\n\n'
        "def upgrade() -> None:\n" + ("\n".join(creates) or "    pass") + "\n\n\n"
        "def downgrade() -> None:\n" + ("\n".join(drops) or "    pass") + "\n"
    )


def _cache_py(p: ServiceProfile) -> str:
    return render(
        '''
        """Redis cache for the {{display}}: {{usage}}."""

        from __future__ import annotations

        import json
        from typing import Any

        import redis

        from novacart_common.logging import get_logger
        from {{pkg}}.config import settings

        logger = get_logger(__name__)

        CACHE_PREFIX = "{{prefix}}"
        CACHE_VERSION = 3


        class {{cls}}:
            def __init__(self, client: redis.Redis | None = None, ttl_seconds: int | None = None) -> None:
                self._redis = client or redis.Redis.from_url(settings.redis_url, socket_timeout=0.25)
                self._ttl_seconds = ttl_seconds or settings.cache_ttl_seconds

            def _key(self, entity_id: str) -> str:
                return f"{CACHE_PREFIX}:v{CACHE_VERSION}:{entity_id}"

            def get(self, entity_id: str) -> dict[str, Any] | None:
                raw = self._redis.get(self._key(entity_id))
                if raw is None:
                    logger.debug("cache.miss", extra={"entity_id": entity_id})
                    return None
                return json.loads(raw)

            def set(self, entity_id: str, value: dict[str, Any]) -> None:
                payload = json.dumps(value, separators=(",", ":"), default=str)
                self._redis.set(self._key(entity_id), payload, ex=self._ttl_seconds)

            def invalidate(self, entity_id: str) -> None:
                self._redis.delete(self._key(entity_id))
                logger.info("cache.invalidated", extra={"entity_id": entity_id})
        ''',
        display=p.display_name,
        usage=p.redis_usage,
        pkg=p.package,
        prefix=p.short.lower(),
        cls=f"{p.short}Cache",
    )


def _consumer_py(p: ServiceProfile) -> str:
    handlers = []
    for topic in p.consumes:
        fields = EVENT_FIELDS[topic]
        extras = ", ".join(f'"{name}": event["{name}"]' for name in fields)
        handlers.append(
            f"async def {handler_name(topic)}(event: dict[str, Any]) -> None:\n"
            f'    logger.info(\n        "event.processed",\n'
            f'        extra={{"topic": "{topic}", "event_id": event["event_id"], {extras}}},\n    )\n'
        )
    handler_map = "\n".join(f'    "{topic}": {handler_name(topic)},' for topic in p.consumes)
    return (
        f'"""Kafka consumer for the {p.display_name}.\n\n'
        f"Subscribes to: {', '.join(p.consumes)}. Each poll returns up to\n"
        "``kafka_max_poll_records`` messages which are processed concurrently; offsets are\n"
        'committed after the whole batch succeeds (at-least-once delivery)."""\n\n'
        "from __future__ import annotations\n\n"
        "import asyncio\nimport json\nfrom collections.abc import Awaitable, Callable\nfrom typing import Any\n\n"
        "from confluent_kafka import Consumer, KafkaError\n\n"
        "from novacart_common.logging import get_logger\n"
        f"from {p.package}.config import settings\n\n"
        "logger = get_logger(__name__)\n\n"
        f"TOPICS = {list(p.consumes)!r}\n\n\n" + "\n\n".join(handlers) + "\n\n"
        "HANDLERS: dict[str, Callable[[dict[str, Any]], Awaitable[None]]] = {\n"
        + handler_map
        + "\n}\n\n\n"
        "class EventConsumer:\n"
        "    def __init__(self) -> None:\n"
        "        self._consumer = Consumer(\n"
        "            {\n"
        '                "bootstrap.servers": settings.kafka_bootstrap_servers,\n'
        '                "group.id": settings.kafka_consumer_group,\n'
        '                "enable.auto.commit": False,\n'
        '                "max.poll.interval.ms": 300000,\n'
        '                "session.timeout.ms": 45000,\n'
        "            }\n"
        "        )\n"
        f"        self._concurrency = {p.config.consumer_concurrency}\n"
        "        self._running = False\n\n"
        "    async def start(self) -> None:\n"
        "        self._consumer.subscribe(TOPICS)\n"
        "        self._running = True\n"
        "        while self._running:\n"
        "            messages = self._consumer.consume(\n"
        "                num_messages=settings.kafka_max_poll_records, timeout=1.0\n"
        "            )\n"
        "            if not messages:\n"
        "                continue\n"
        "            await self._process_batch(messages)\n"
        "            self._consumer.commit(asynchronous=False)\n\n"
        "    async def _process_batch(self, messages: list[Any]) -> None:\n"
        "        semaphore = asyncio.Semaphore(self._concurrency)\n\n"
        "        async def process(message: Any) -> None:\n"
        "            async with semaphore:\n"
        "                await self._handle(message)\n\n"
        "        await asyncio.gather(*(process(message) for message in messages))\n\n"
        "    async def _handle(self, message: Any) -> None:\n"
        "        if message.error():\n"
        "            if message.error().code() != KafkaError._PARTITION_EOF:\n"
        '                logger.error("consumer.message_error", extra={"error": str(message.error())})\n'
        "            return\n"
        "        handler = HANDLERS.get(message.topic())\n"
        "        if handler is None:\n"
        '            logger.warning("consumer.unhandled_topic", extra={"topic": message.topic()})\n'
        "            return\n"
        "        await handler(json.loads(message.value()))\n\n"
        "    def stop(self) -> None:\n"
        "        self._running = False\n"
        "        self._consumer.close()\n"
    )


def _event_schemas_py(p: ServiceProfile) -> str:
    classes = []
    for topic in p.produces:
        fields = "\n".join(f"    {name}: {_event_field_type(name)}" for name in EVENT_FIELDS[topic])
        classes.append(
            f"@dataclass(frozen=True)\nclass {event_class(topic)}:\n"
            f'    """Published to ``{topic}``."""\n\n'
            f'    topic: ClassVar[str] = "{topic}"\n\n'
            f"{fields}\n"
            "    event_id: str = field(default_factory=lambda: uuid4().hex)\n"
            "    occurred_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())\n"
            "    schema_version: int = SCHEMA_VERSION\n\n"
            "    def to_message(self) -> dict[str, object]:\n"
            "        return asdict(self)\n"
        )
    return (
        f'"""Event payloads published by the {p.display_name} (JSON over Kafka).\n\n'
        "Consumers depend on these field names: renaming a field is a breaking change\n"
        'and requires a new schema version plus a dual-publish period."""\n\n'
        "from __future__ import annotations\n\n"
        "from dataclasses import asdict, dataclass, field\n"
        "from datetime import UTC, datetime\n"
        "from typing import ClassVar\n"
        "from uuid import uuid4\n\n"
        "SCHEMA_VERSION = 2\n\n\n" + "\n\n".join(classes)
    )


def _producer_py(p: ServiceProfile) -> str:
    return render(
        '''
        """Publishes {{display}} domain events to Kafka."""

        from __future__ import annotations

        import json
        from typing import Any

        from confluent_kafka import Producer

        from novacart_common.logging import get_logger
        from {{pkg}}.config import settings

        logger = get_logger(__name__)


        class EventPublisher:
            def __init__(self) -> None:
                self._producer = Producer(
                    {
                        "bootstrap.servers": settings.kafka_bootstrap_servers,
                        "enable.idempotence": True,
                        "acks": "all",
                        "linger.ms": 5,
                    }
                )

            def publish(self, event: Any) -> None:
                payload = event.to_message()
                self._producer.produce(
                    event.topic,
                    key=str(payload.get("event_id")),
                    value=json.dumps(payload).encode("utf-8"),
                    on_delivery=self._on_delivery,
                )
                self._producer.poll(0)

            def flush(self, timeout: float = 5.0) -> None:
                self._producer.flush(timeout)

            @staticmethod
            def _on_delivery(error: Any, message: Any) -> None:
                if error is not None:
                    logger.error("event.delivery_failed", extra={"topic": message.topic(), "error": str(error)})
        ''',
        display=p.display_name,
        pkg=p.package,
    )


def _internal_client_py(p: ServiceProfile, target: str, purpose: str) -> str:
    method, verb, path, params = CLIENT_METHODS[target]
    target_profile = SERVICES_BY_ID[target]
    signature = ", ".join(
        f"{name}: {'list[dict[str, Any]]' if name == 'lines' else 'str'}" for name in params
    )
    path_params = set(re.findall(r"\{(\w+)\}", path))
    body = [name for name in params if name not in path_params]
    url = f'f"{path}"' if path_params else f'"{path}"'
    fields = "{" + ", ".join(f'"{name}": {name}' for name in body) + "}"
    if verb == "get":
        call = (
            f"await self._http.get({url}, params={fields})"
            if body
            else f"await self._http.get({url})"
        )
    else:
        call = f"await self._http.post({url}, json={fields})"
    return (
        f'"""HTTP client for the {target_profile.display_name} ({purpose})."""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
        "from novacart_common.http import ResilientHttpClient\n"
        f"from {p.package}.config import settings\n\n\n"
        f"class {client_class(target)}:\n"
        "    def __init__(self) -> None:\n"
        "        self._http = ResilientHttpClient(\n"
        f"            base_url=settings.{dependency_url_setting(target)},\n"
        "            timeout_seconds=settings.http_timeout_seconds,\n"
        "            max_retries=settings.http_max_retries,\n"
        "            circuit_breaker_threshold=0.5,\n"
        "        )\n\n"
        f"    async def {method}(self{', ' + signature if signature else ''}) -> dict[str, Any]:\n"
        f"        return {call}\n"
    )


def _external_client_py(p: ServiceProfile, provider: ExternalProvider) -> str:
    return render(
        '''
        """Client for {{name}} ({{purpose}})."""

        from __future__ import annotations

        from typing import Any

        from novacart_common.http import ResilientHttpClient, UpstreamUnavailable
        from {{pkg}}.config import settings


        class ProviderUnavailable(UpstreamUnavailable):
            """{{name}} could not be reached or returned 5xx after retries."""


        class {{cls}}:
            def __init__(self) -> None:
                self._http = ResilientHttpClient(
                    base_url=settings.{{prefix}}_base_url,
                    timeout_seconds=settings.http_timeout_seconds,
                    max_retries=settings.http_max_retries,
                    backoff_seconds=0.2,
                    headers={"Authorization": f"Bearer {settings.{{prefix}}_api_key}"},
                )

            async def call(self, operation: str, **payload: Any) -> dict[str, Any]:
                try:
                    return await self._http.post(f"/{operation}", json=payload)
                except UpstreamUnavailable as exc:
                    raise ProviderUnavailable(str(exc)) from exc
        ''',
        name=provider.name,
        purpose=provider.purpose,
        pkg=p.package,
        cls=provider.client_class,
        prefix=provider.settings_prefix,
    )


def _requirements_txt(p: ServiceProfile) -> str:
    lines = [
        "fastapi==0.115.6",
        "uvicorn[standard]==0.32.1",
        "pydantic-settings==2.6.1",
        "httpx==0.28.1",
    ]
    if p.postgres_db:
        lines += ["sqlalchemy==2.0.36", "psycopg[binary]==3.2.3", "alembic==1.14.0"]
    if p.redis_cluster:
        lines.append("redis==5.2.1")
    if p.produces or p.consumes:
        lines.append("confluent-kafka==2.6.1")
    if "opensearch:search-products" in p.extra_datastores:
        lines.append("opensearch-py==2.7.1")
    if p.id in {"auth-service", "api-gateway"}:
        lines.append("pyjwt[crypto]==2.10.1")
    lines += ["novacart-common==1.12.0", "", "# test", "pytest==8.3.4", "pytest-asyncio==0.24.0"]
    return "\n".join(lines) + "\n"


def _readme(p: ServiceProfile) -> str:
    endpoints = markdown_table(
        ["Method", "Path", "Description"],
        [[ep.method, f"`{ep.path}`", ep.summary] for ep in p.endpoints],
    )
    deps = [f"`{d.service}` ({d.criticality.value}): {d.purpose}" for d in p.http_dependencies]
    deps += [f"{x.name} (external): {x.purpose}" for x in p.external]
    deps += [f"Kafka `{t}` from `{TOPIC_PRODUCERS[t]}`" for t in p.consumes]
    return render(
        """
        # {{display}}

        {{description}}

        Owner: **{{team}}** · On-call: `{{oncall}}` · Tier: `{{tier}}` · Port: {{port}}

        ## Endpoints

        {{endpoints}}

        ## Dependencies

        {{deps}}

        ## Datastores

        {{stores}}

        ## Running locally

        ```bash
        cd services/{{id}}
        pip install -r requirements.txt
        uvicorn {{pkg}}.main:app --port {{port}} --reload
        pytest
        ```

        Configuration is read from environment variables prefixed `{{prefix}}_`
        (see `{{pkg}}/config.py`); production values are in `deploy/{{id}}.yaml`.
        """,
        display=p.display_name,
        description=p.description,
        team=p.team,
        oncall=p.oncall_channel,
        tier=p.tier.value,
        port=p.port,
        endpoints=endpoints,
        deps=bullet_list(deps) or "- none",
        stores=bullet_list([f"`{s}`" for s in p.datastores]) or "- none (stateless)",
        id=p.id,
        pkg=p.package,
        prefix=p.env_prefix,
    )


def _tests(p: ServiceProfile) -> list[tuple[str, str]]:
    module, cls = primary_class(p.id)
    first = p.endpoints[0]
    route_test = render(
        """
        from unittest.mock import AsyncMock

        from fastapi.testclient import TestClient

        from {{pkg}}.api.routes import get_service
        from {{pkg}}.main import app


        def test_healthz() -> None:
            with TestClient(app) as client:
                assert client.get("/healthz").json()["status"] == "ok"


        def test_{{handler}}_delegates_to_service() -> None:
            service = AsyncMock()
            service.{{handler}}.return_value = {"ok": True}
            app.dependency_overrides[get_service] = lambda: service
            try:
                with TestClient(app) as client:
                    response = client.{{verb}}("{{path}}"{{body}})
                assert response.status_code == 200
                service.{{handler}}.assert_awaited_once()
            finally:
                app.dependency_overrides.clear()
        """,
        pkg=p.package,
        handler=first.handler,
        verb=first.method.lower(),
        path=re.sub(r"\{(\w+)\}", "test-id", first.path),
        body=", json={}" if first.method in {"POST", "PUT", "PATCH"} else "",
    )
    domain_test = render(
        """
        import inspect

        from {{pkg}}.{{module}} import {{cls}}


        def test_public_handlers_are_coroutines() -> None:
            for name in {{handlers}}:
                assert inspect.iscoroutinefunction(getattr({{cls}}, name)), name


        def test_can_be_built_from_settings() -> None:
            assert callable({{cls}}.from_settings)
        """,
        pkg=p.package,
        module=module,
        cls=cls,
        handlers=[ep.handler for ep in p.endpoints],
    )
    tests = [("tests/test_routes.py", route_test), (f"tests/test_{module}.py", domain_test)]
    if p.postgres_db:
        model = TABLE_MODELS[primary_table(p)][0]
        tests.append(
            (
                "tests/test_repository.py",
                render(
                    """
                from unittest.mock import MagicMock, patch

                from {{pkg}}.db.repository import {{model}}Repository


                def test_bulk_update_locks_rows_in_primary_key_order() -> None:
                    session = MagicMock()
                    session.execute.return_value.rowcount = 1
                    with patch("{{pkg}}.db.repository.session_scope") as scope:
                        scope.return_value.__enter__.return_value = session
                        updated = {{model}}Repository().bulk_update_status(["c", "a", "b"], "archived")
                    assert updated == 3
                    locked = [call.args[0].whereclause.right.value for call in session.execute.call_args_list[::2]]
                    assert locked == ["a", "b", "c"]
                """,
                    pkg=p.package,
                    model=model,
                ),
            )
        )
    if p.redis_cluster:
        tests.append(
            (
                "tests/test_cache.py",
                render(
                    """
                from unittest.mock import MagicMock

                from {{pkg}}.cache import {{cls}}


                def test_set_always_applies_ttl() -> None:
                    client = MagicMock()
                    {{cls}}(client=client, ttl_seconds=60).set("abc", {"x": 1})
                    assert client.set.call_args.kwargs["ex"] == 60


                def test_invalidate_deletes_key() -> None:
                    client = MagicMock()
                    {{cls}}(client=client, ttl_seconds=60).invalidate("abc")
                    client.delete.assert_called_once()
                """,
                    pkg=p.package,
                    cls=f"{p.short}Cache",
                ),
            )
        )
    return tests


def service_files(p: ServiceProfile) -> list[RepoFile]:
    root, pkg = p.repo_path, f"{p.repo_path}/{p.package}"
    level = p.access_level

    def src(path: str, content: str, kind: CodeFileKind = CodeFileKind.SOURCE) -> RepoFile:
        return RepoFile(path, p.id, "python", kind, content, level)

    files = [
        RepoFile(f"{root}/README.md", p.id, "markdown", CodeFileKind.DOCS, _readme(p)),
        RepoFile(
            f"{root}/requirements.txt", p.id, "text", CodeFileKind.BUILD, _requirements_txt(p)
        ),
        RepoFile(
            f"{root}/deploy/{p.id}.yaml", p.id, "yaml", CodeFileKind.CONFIG, _deploy_yaml(p), level
        ),
        src(f"{pkg}/__init__.py", f'"""{p.display_name}: {p.description}"""\n'),
        src(f"{pkg}/main.py", _main_py(p)),
        src(f"{pkg}/config.py", _config_py(p), CodeFileKind.CONFIG),
        src(f"{pkg}/api/__init__.py", '"""HTTP API."""\n'),
        src(f"{pkg}/api/routes.py", _routes_py(p)),
    ]
    for module in p.domain_modules:
        files.append(src(f"{pkg}/{module}.py", domain_module_source(p.id, module)))
    if p.postgres_db:
        files += [
            src(f"{pkg}/db/__init__.py", '"""Persistence."""\n'),
            src(f"{pkg}/db/database.py", _database_py(p)),
            src(f"{pkg}/db/models.py", _models_py(p)),
            src(f"{pkg}/db/repository.py", _repository_py(p)),
            src(
                f"{root}/migrations/versions/0001_initial.py",
                _migration_initial(p),
                CodeFileKind.MIGRATION,
            ),
            src(
                f"{root}/migrations/versions/0002_add_indexes.py",
                _migration_indexes(p),
                CodeFileKind.MIGRATION,
            ),
        ]
    if p.redis_cluster:
        files.append(src(f"{pkg}/cache.py", _cache_py(p)))
    if p.produces or p.consumes:
        files.append(src(f"{pkg}/events/__init__.py", '"""Kafka integration."""\n'))
    if p.consumes:
        files.append(src(f"{pkg}/events/consumer.py", _consumer_py(p)))
    if p.produces:
        files += [
            src(f"{pkg}/events/producer.py", _producer_py(p)),
            src(f"{pkg}/events/schemas.py", _event_schemas_py(p)),
        ]
    if has_http_clients(p):
        files.append(src(f"{pkg}/clients/__init__.py", '"""Outbound HTTP clients."""\n'))
        for dep in p.http_dependencies:
            files.append(
                src(
                    f"{pkg}/clients/{client_module(dep.service)}.py",
                    _internal_client_py(p, dep.service, dep.purpose),
                )
            )
        for provider in p.external:
            files.append(
                src(f"{pkg}/clients/{provider.client_module}.py", _external_client_py(p, provider))
            )
    for path, content in _tests(p):
        files.append(RepoFile(f"{root}/{path}", p.id, "python", CodeFileKind.TEST, content))
    return files


# --- Shared library and repository root ----------------------------------------------------


def _shared_files() -> list[RepoFile]:
    lib = "libs/novacart_common"
    files = {
        f"{lib}/README.md": (
            "markdown",
            CodeFileKind.DOCS,
            "# novacart-common\n\nShared building blocks used by every NovaCart service: structured logging,\n"
            "request tracing, a resilient HTTP client (timeouts, retries with jittered backoff,\n"
            "circuit breaker) and id generation. Versioned and released independently;\n"
            "services pin it in their requirements.\n",
        ),
        f"{lib}/novacart_common/__init__.py": (
            "python",
            CodeFileKind.SOURCE,
            '"""Shared NovaCart service library."""\n\n__version__ = "1.12.0"\n',
        ),
        f"{lib}/novacart_common/ids.py": (
            "python",
            CodeFileKind.SOURCE,
            'from uuid import uuid4\n\n\ndef new_id() -> str:\n    """Random UUID4 string used as primary key for new records."""\n    return str(uuid4())\n',
        ),
        f"{lib}/novacart_common/logging.py": (
            "python",
            CodeFileKind.SOURCE,
            render(
                '''
            """JSON logging with the fields every NovaCart log line must carry."""

            from __future__ import annotations

            import json
            import logging
            import sys

            from novacart_common.tracing import current_trace


            class JsonFormatter(logging.Formatter):
                def __init__(self, service: str) -> None:
                    super().__init__()
                    self._service = service

                def format(self, record: logging.LogRecord) -> str:
                    trace = current_trace()
                    payload = {
                        "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                        "service": self._service,
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "trace_id": trace.trace_id if trace else None,
                        "span_id": trace.span_id if trace else None,
                    }
                    payload.update(getattr(record, "extra_fields", {}))
                    return json.dumps(payload, default=str)


            def configure_logging(service: str, level: str = "INFO") -> None:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(JsonFormatter(service))
                root = logging.getLogger()
                root.handlers[:] = [handler]
                root.setLevel(level)


            def get_logger(name: str) -> logging.Logger:
                return logging.getLogger(name)
            '''
            ),
        ),
        f"{lib}/novacart_common/tracing.py": (
            "python",
            CodeFileKind.SOURCE,
            render(
                '''
            """W3C trace-context propagation (traceparent header)."""

            from __future__ import annotations

            import secrets
            from contextvars import ContextVar
            from dataclasses import dataclass


            @dataclass(frozen=True)
            class TraceContext:
                trace_id: str
                span_id: str


            _current: ContextVar[TraceContext | None] = ContextVar("trace", default=None)


            def new_request_id() -> str:
                return secrets.token_hex(16)


            def start_trace(traceparent: str | None = None) -> TraceContext:
                trace_id = traceparent.split("-")[1] if traceparent else secrets.token_hex(16)
                context = TraceContext(trace_id=trace_id, span_id=secrets.token_hex(8))
                _current.set(context)
                return context


            def current_trace() -> TraceContext | None:
                return _current.get()


            def traceparent_header() -> dict[str, str]:
                context = _current.get()
                if context is None:
                    return {}
                return {"traceparent": f"00-{context.trace_id}-{context.span_id}-01"}
            '''
            ),
        ),
        f"{lib}/novacart_common/circuit_breaker.py": (
            "python",
            CodeFileKind.SOURCE,
            render(
                '''
            """Rolling-window circuit breaker."""

            from __future__ import annotations

            import time
            from collections import deque


            class CircuitOpenError(RuntimeError):
                pass


            class CircuitBreaker:
                """Opens when the failure ratio over the last ``window`` calls exceeds
                ``threshold``; after ``cooldown_seconds`` one trial call is allowed."""

                def __init__(self, threshold: float = 0.5, window: int = 50, cooldown_seconds: float = 10.0) -> None:
                    self._threshold = threshold
                    self._results: deque[bool] = deque(maxlen=window)
                    self._cooldown = cooldown_seconds
                    self._opened_at: float | None = None

                def before_call(self) -> None:
                    if self._opened_at is None:
                        return
                    if time.monotonic() - self._opened_at < self._cooldown:
                        raise CircuitOpenError("circuit open")
                    self._opened_at = None  # half-open: allow a trial call

                def record(self, success: bool) -> None:
                    self._results.append(success)
                    failures = self._results.count(False)
                    if len(self._results) >= 10 and failures / len(self._results) > self._threshold:
                        self._opened_at = time.monotonic()
            '''
            ),
        ),
        f"{lib}/novacart_common/http.py": (
            "python",
            CodeFileKind.SOURCE,
            render(
                '''
            """HTTP client with timeouts, bounded retries and a circuit breaker.

            Retries apply only to idempotent failures (connect errors, timeouts, 502/503/504)
            and use exponential backoff with full jitter: delay = random(0, backoff * 2**attempt).
            A per-call total budget is timeout_seconds * (max_retries + 1).
            """

            from __future__ import annotations

            import asyncio
            import random
            from typing import Any

            import httpx

            from novacart_common.circuit_breaker import CircuitBreaker
            from novacart_common.tracing import traceparent_header

            RETRYABLE_STATUS = frozenset({502, 503, 504})


            class UpstreamUnavailable(RuntimeError):
                pass


            class ResilientHttpClient:
                def __init__(
                    self,
                    base_url: str,
                    timeout_seconds: float,
                    max_retries: int = 2,
                    backoff_seconds: float = 0.1,
                    circuit_breaker_threshold: float = 0.5,
                    headers: dict[str, str] | None = None,
                ) -> None:
                    self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, headers=headers)
                    self._max_retries = max_retries
                    self._backoff = backoff_seconds
                    self._breaker = CircuitBreaker(threshold=circuit_breaker_threshold)

                async def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
                    return await self._request("GET", path, params=params)

                async def post(self, path: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
                    return await self._request("POST", path, json=json)

                async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
                    self._breaker.before_call()
                    last_error: Exception | None = None
                    for attempt in range(self._max_retries + 1):
                        try:
                            response = await self._client.request(method, path, headers=traceparent_header(), **kwargs)
                            if response.status_code in RETRYABLE_STATUS:
                                raise httpx.HTTPStatusError("retryable", request=response.request, response=response)
                            response.raise_for_status()
                            self._breaker.record(True)
                            return response.json()
                        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                            last_error = exc
                            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in RETRYABLE_STATUS:
                                break
                            if attempt < self._max_retries:
                                await asyncio.sleep(random.uniform(0, self._backoff * 2**attempt))
                    self._breaker.record(False)
                    raise UpstreamUnavailable(f"{method} {path} failed: {last_error}")
            '''
            ),
        ),
        f"{lib}/tests/test_circuit_breaker.py": (
            "python",
            CodeFileKind.TEST,
            render(
                """
            import pytest

            from novacart_common.circuit_breaker import CircuitBreaker, CircuitOpenError


            def test_opens_after_failure_ratio_exceeded() -> None:
                breaker = CircuitBreaker(threshold=0.5, window=10, cooldown_seconds=60)
                for _ in range(10):
                    breaker.record(False)
                with pytest.raises(CircuitOpenError):
                    breaker.before_call()


            def test_stays_closed_when_healthy() -> None:
                breaker = CircuitBreaker(threshold=0.5, window=10)
                for _ in range(10):
                    breaker.record(True)
                breaker.before_call()
            """
            ),
        ),
        f"{lib}/pyproject.toml": (
            "toml",
            CodeFileKind.BUILD,
            '[project]\nname = "novacart-common"\nversion = "1.12.0"\nrequires-python = ">=3.11"\ndependencies = ["httpx>=0.27"]\n',
        ),
    }
    return [
        RepoFile(path, None, language, kind, content)
        for path, (language, kind, content) in files.items()
    ]


def _root_files() -> list[RepoFile]:
    services = markdown_table(
        ["Service", "Team", "Tier", "Description"],
        [[f"[`{s.id}`](services/{s.id})", s.team, s.tier.value, s.description] for s in SERVICES],
    )
    readme = render(
        """
        # NovaCart monorepo

        Backend services for the NovaCart e-commerce platform. Each service is a FastAPI
        application under `services/<name>` with its own deploy manifest; shared code lives
        in `libs/novacart_common`.

        {{services}}

        ## Conventions

        - Configuration only via environment variables (`<SERVICE>_*`); secrets from Vault.
        - Every outbound HTTP call goes through `novacart_common.http.ResilientHttpClient`.
        - Database schema changes are Alembic migrations under `services/<name>/migrations`.
        - Kafka events are versioned dataclasses in `<package>/events/schemas.py`.
        - Deploys: canary for tier_0 services (5% -> 25% -> 100%), rolling otherwise.
        """,
        services=services,
    )
    topics = "\n".join(
        f"  - name: {topic}\n    partitions: {TOPIC_PARTITIONS[topic]}\n    producer: {producer}\n"
        f"    retention_hours: 168"
        for topic, producer in sorted(TOPIC_PRODUCERS.items())
    )
    redis = "\n".join(
        f"  - name: {s.redis_cluster}\n    owner: {s.id}\n    maxmemory: {'8gb' if s.id == 'cart-service' else '2gb'}\n"
        f"    maxmemory_policy: {'volatile-lru' if s.id in {'cart-service', 'payment-service'} else 'allkeys-lru'}"
        for s in SERVICES
        if s.redis_cluster
    )
    ci = render(
        """
        name: ci
        on:
          pull_request:
          push:
            branches: [main]
        jobs:
          test:
            runs-on: ubuntu-latest
            strategy:
              matrix:
                service: [{{services}}]
            steps:
              - uses: actions/checkout@v4
              - uses: actions/setup-python@v5
                with:
                  python-version: "3.11"
              - run: pip install -r services/${{ matrix.service }}/requirements.txt -e libs/novacart_common
              - run: pytest services/${{ matrix.service }}/tests
        """,
        services=", ".join(s.id for s in SERVICES),
    )
    codeowners = "\n".join(f"/services/{s.id}/ @novacart/{s.team}" for s in SERVICES)
    return [
        RepoFile("README.md", None, "markdown", CodeFileKind.DOCS, readme),
        RepoFile(".github/workflows/ci.yml", None, "yaml", CodeFileKind.BUILD, ci),
        RepoFile(
            "CODEOWNERS",
            None,
            "text",
            CodeFileKind.BUILD,
            codeowners + "\n/libs/ @novacart/platform\n",
        ),
        RepoFile(
            "deploy/platform/kafka-topics.yaml",
            None,
            "yaml",
            CodeFileKind.CONFIG,
            "# Kafka topics (managed by the platform team)\ntopics:\n" + topics + "\n",
        ),
        RepoFile(
            "deploy/platform/redis-clusters.yaml",
            None,
            "yaml",
            CodeFileKind.CONFIG,
            "# Redis clusters (managed by the platform team)\nclusters:\n" + redis + "\n",
        ),
    ]


def build_repository() -> dict[str, RepoFile]:
    files: list[RepoFile] = _root_files() + _shared_files()
    for profile in SERVICES:
        files.extend(service_files(profile))
    by_path = {f.path: f for f in files}
    if len(by_path) != len(files):
        raise ValueError("duplicate repository paths")
    return dict(sorted(by_path.items()))
