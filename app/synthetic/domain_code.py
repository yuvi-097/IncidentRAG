"""Hand-written domain modules for each NovaCart service, plus the code-level
faults that deployments can introduce into them.

Fault specs name exact snippets of these sources; ``changes.py`` turns them into
real unified diffs and tests assert every snippet matches exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.synthetic.text import render


@dataclass(frozen=True)
class CodeFault:
    """A plausible-looking change to a hand-written module that breaks production."""

    service_id: str
    module: str
    replacements: tuple[tuple[str, str], ...]  # (HEAD snippet, faulty snippet)
    pr_title: str
    pr_rationale: str  # what the author believed the change did
    root_cause: str  # what it actually did
    fix_title: str
    symptom: str
    exception: str
    error_message: str
    endpoint: str  # "POST /v1/orders"
    status_code: int


# (service, module) -> (primary class name or None, source)
_SOURCES: dict[tuple[str, str], str] = {}
_PRIMARY: dict[str, tuple[str, str]] = {}


def _module(service_id: str, module: str, source: str, primary: str | None = None) -> None:
    _SOURCES[(service_id, module)] = render(source)
    if primary:
        _PRIMARY[service_id] = (module, primary)


def domain_module_source(service_id: str, module: str) -> str:
    return _SOURCES[(service_id, module)]


def primary_class(service_id: str) -> tuple[str, str]:
    """(module, class) that implements the service's HTTP handlers."""
    return _PRIMARY[service_id]


# ----------------------------------------------------------------------------- api-gateway

_module(
    "api-gateway",
    "routing",
    '''
    """Routes public API requests to backend services."""

    from __future__ import annotations

    from dataclasses import dataclass
    from typing import Any

    import httpx

    from novacart_common.logging import get_logger
    from novacart_common.tracing import traceparent_header

    logger = get_logger(__name__)


    @dataclass(frozen=True)
    class Upstream:
        name: str
        base_url: str
        timeout_seconds: float


    UPSTREAMS: dict[str, Upstream] = {
        "auth": Upstream("auth-service", "http://auth-service.novacart.svc.cluster.local:8081", 2.0),
        "users": Upstream("user-service", "http://user-service.novacart.svc.cluster.local:8082", 2.0),
        "products": Upstream("product-service", "http://product-service.novacart.svc.cluster.local:8083", 2.0),
        "carts": Upstream("cart-service", "http://cart-service.novacart.svc.cluster.local:8085", 2.0),
        "orders": Upstream("order-service", "http://order-service.novacart.svc.cluster.local:8086", 5.0),
        "payments": Upstream("payment-service", "http://payment-service.novacart.svc.cluster.local:8087", 5.0),
        "search": Upstream("search-service", "http://search-service.novacart.svc.cluster.local:8090", 1.5),
        "recommendations": Upstream(
            "recommendation-service", "http://recommendation-service.novacart.svc.cluster.local:8089", 0.8
        ),
    }


    class UnknownRoute(LookupError):
        pass


    class Router:
        def __init__(self, client: httpx.AsyncClient) -> None:
            self._client = client

        @classmethod
        def from_settings(cls) -> Router:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            return cls(httpx.AsyncClient(limits=limits))

        async def proxy_get(self, service: str, path: str) -> dict[str, Any]:
            return await self._forward("GET", service, path)

        async def proxy_post(
            self, service: str, path: str, payload: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            return await self._forward("POST", service, path, json=payload or {})

        async def _forward(self, method: str, service: str, path: str, **kwargs: Any) -> dict[str, Any]:
            upstream = UPSTREAMS.get(service)
            if upstream is None:
                raise UnknownRoute(service)
            url = f"{upstream.base_url}/v1/{service}/{path}"
            response = await self._client.request(
                method, url, timeout=upstream.timeout_seconds, headers=traceparent_header(), **kwargs
            )
            if response.status_code >= 500:
                logger.error(
                    "gateway.upstream_error",
                    extra={"upstream": upstream.name, "status": response.status_code, "path": path},
                )
            response.raise_for_status()
            return response.json()
''',
    primary="Router",
)

_module(
    "api-gateway",
    "rate_limiter",
    '''
    """Token-bucket rate limiting per API client, backed by Redis."""

    from __future__ import annotations

    import time

    import redis

    from api_gateway.config import settings
    from novacart_common.logging import get_logger

    logger = get_logger(__name__)

    DEFAULT_LIMIT_RPS = 50
    DEFAULT_BURST = 100
    PARTNER_LIMITS = {"partner-marketplace": 400, "partner-affiliates": 200}

    # Atomic refill-and-take. Returns 1 when the request is allowed.
    TOKEN_BUCKET_LUA = """
    local capacity = tonumber(ARGV[2])
    local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens') or capacity)
    local last = tonumber(redis.call('HGET', KEYS[1], 'ts') or ARGV[3])
    tokens = math.min(capacity, tokens + (tonumber(ARGV[3]) - last) * tonumber(ARGV[1]))
    local allowed = tokens >= 1 and 1 or 0
    if allowed == 1 then tokens = tokens - 1 end
    redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', ARGV[3])
    redis.call('EXPIRE', KEYS[1], 60)
    return allowed
    """


    class RateLimitExceeded(Exception):
        def __init__(self, retry_after: float) -> None:
            super().__init__("rate limit exceeded")
            self.retry_after = retry_after


    class TokenBucketRateLimiter:
        def __init__(self, client: redis.Redis | None = None) -> None:
            self._redis = client or redis.Redis.from_url(settings.redis_url, socket_timeout=0.05)
            self._script = self._redis.register_script(TOKEN_BUCKET_LUA)

        def check(self, client_id: str, client_ip: str) -> None:
            limit = PARTNER_LIMITS.get(client_id, DEFAULT_LIMIT_RPS)
            key = f"ratelimit:{client_id}"
            try:
                allowed = self._script(keys=[key], args=[limit, DEFAULT_BURST, time.time()])
            except redis.RedisError:
                # Fail open: never reject traffic because the limiter itself is unavailable.
                logger.warning("ratelimit.redis_unavailable", extra={"client_id": client_id})
                return
            if not allowed:
                logger.info("ratelimit.rejected", extra={"client_id": client_id, "limit_rps": limit})
                raise RateLimitExceeded(retry_after=1.0)
''',
)

_module(
    "api-gateway",
    "auth_middleware",
    '''
    """Bearer-token authentication applied to every proxied request."""

    from __future__ import annotations

    import time
    from typing import Any

    import httpx
    import jwt

    from novacart_common.logging import get_logger

    logger = get_logger(__name__)

    JWKS_URL = "http://auth-service.novacart.svc.cluster.local:8081/v1/auth/.well-known/jwks.json"
    JWKS_CACHE_TTL_SECONDS = 300
    CLOCK_SKEW_LEEWAY_SECONDS = 30
    ISSUER = "https://auth.novacart.example"
    PUBLIC_PATHS = ("/healthz", "/v1/auth/token", "/v1/auth/refresh", "/v1/products", "/v1/search")


    class AuthenticationError(Exception):
        pass


    class JwksCache:
        """Caches auth-service signing keys; refetches on expiry or an unknown ``kid``
        (signing keys rotate weekly with a 24h overlap)."""

        def __init__(self, client: httpx.Client | None = None) -> None:
            self._client = client or httpx.Client(timeout=1.5)
            self._keys: dict[str, Any] = {}
            self._fetched_at = 0.0

        def _refresh(self) -> None:
            response = self._client.get(JWKS_URL)
            response.raise_for_status()
            self._keys = {
                jwk["kid"]: jwt.PyJWK(jwk).key for jwk in response.json()["keys"]
            }
            self._fetched_at = time.monotonic()
            logger.info("jwks.refreshed", extra={"kids": sorted(self._keys)})

        def get_key(self, kid: str) -> Any:
            expired = time.monotonic() - self._fetched_at > JWKS_CACHE_TTL_SECONDS
            if expired or kid not in self._keys:
                self._refresh()
            if kid not in self._keys:
                raise AuthenticationError(f"unknown signing key: {kid}")
            return self._keys[kid]


    def authenticate(authorization: str | None, jwks: JwksCache) -> dict[str, Any]:
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("missing bearer token")
        token = authorization.removeprefix("Bearer ")
        kid = jwt.get_unverified_header(token)["kid"]
        try:
            return jwt.decode(
                token, jwks.get_key(kid), algorithms=["RS256"], issuer=ISSUER, leeway=CLOCK_SKEW_LEEWAY_SECONDS
            )
        except jwt.PyJWTError as exc:
            logger.warning("auth.token_rejected", extra={"kid": kid, "reason": str(exc)})
            raise AuthenticationError(str(exc)) from exc
''',
)

# ----------------------------------------------------------------------------- auth-service

_module(
    "auth-service",
    "tokens",
    '''
    """Access / refresh token issuance and validation."""

    from __future__ import annotations

    import hashlib
    import secrets
    import time
    from typing import Any

    import jwt

    from auth_service.clients.user_client import UserClient
    from auth_service.db.repository import CredentialRepository
    from auth_service.jwks import SigningKeyRing, verify_password
    from novacart_common.logging import get_logger

    logger = get_logger(__name__)

    ACCESS_TOKEN_TTL_SECONDS = 900
    REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 3600
    CLOCK_SKEW_LEEWAY_SECONDS = 30
    ISSUER = "https://auth.novacart.example"


    class InvalidCredentials(Exception):
        pass


    class TokenService:
        def __init__(self, keys: SigningKeyRing, credentials: CredentialRepository, users: UserClient) -> None:
            self._keys = keys
            self._credentials = credentials
            self._users = users

        @classmethod
        def from_settings(cls) -> TokenService:
            return cls(SigningKeyRing.load(), CredentialRepository(), UserClient())

        async def issue_token(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            request = payload or {}
            credential = self._credentials.get(request.get("username", ""))
            if credential is None or not verify_password(request.get("password", ""), credential["password_hash"]):
                logger.info("auth.login_failed", extra={"username": request.get("username")})
                raise InvalidCredentials()
            user = await self._users.get_user(credential["user_id"])
            if user["status"] != "active":
                raise InvalidCredentials()
            return self._mint(credential["user_id"])

        async def refresh_token(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            token_hash = hashlib.sha256((payload or {})["refresh_token"].encode()).hexdigest()
            record = self._keys.consume_refresh_token(token_hash)
            if record is None:
                raise InvalidCredentials()
            return self._mint(record["user_id"])

        async def introspect_token(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            try:
                claims = self.validate((payload or {})["token"])
            except jwt.PyJWTError as exc:
                return {"active": False, "reason": str(exc)}
            return {"active": True, **claims}

        async def get_jwks(self) -> dict[str, Any]:
            return self._keys.jwks()

        def _mint(self, subject: str) -> dict[str, Any]:
            now = int(time.time())
            key = self._keys.active()
            claims = {"iss": ISSUER, "sub": subject, "iat": now, "nbf": now, "exp": now + ACCESS_TOKEN_TTL_SECONDS}
            access = jwt.encode(claims, key.private_pem, algorithm="RS256", headers={"kid": key.kid})
            refresh = secrets.token_urlsafe(48)
            self._keys.store_refresh_token(subject, hashlib.sha256(refresh.encode()).hexdigest(), REFRESH_TOKEN_TTL_SECONDS)
            return {"access_token": access, "refresh_token": refresh, "token_type": "Bearer", "expires_in": ACCESS_TOKEN_TTL_SECONDS}

        def validate(self, token: str) -> dict[str, Any]:
            kid = jwt.get_unverified_header(token)["kid"]
            return jwt.decode(
                token, self._keys.public_key(kid), algorithms=["RS256"], issuer=ISSUER, leeway=CLOCK_SKEW_LEEWAY_SECONDS
            )
''',
    primary="TokenService",
)

_module(
    "auth-service",
    "jwks",
    '''
    """Signing-key ring: weekly RSA key rotation with a 24h overlap window."""

    from __future__ import annotations

    import hashlib
    import hmac
    from dataclasses import dataclass
    from typing import Any

    from auth_service.cache import AuthCache
    from auth_service.db.database import session_scope
    from auth_service.db.models import SigningKey
    from novacart_common.logging import get_logger

    logger = get_logger(__name__)

    ROTATION_OVERLAP_SECONDS = 24 * 3600


    @dataclass(frozen=True)
    class KeyPair:
        kid: str
        private_pem: str
        public_jwk: dict[str, Any]


    def verify_password(password: str, password_hash: str) -> bool:
        salt, expected = password_hash.split("$", 1)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310_000).hex()
        return hmac.compare_digest(digest, expected)


    class SigningKeyRing:
        def __init__(self, active: KeyPair, published: list[dict[str, Any]], cache: AuthCache) -> None:
            self._active = active
            self._published = published  # active + retiring keys
            self._cache = cache

        @classmethod
        def load(cls) -> SigningKeyRing:
            with session_scope() as session:
                rows = session.query(SigningKey).filter(SigningKey.status.in_(["active", "retiring"])).all()
            active = next(row for row in rows if row.status == "active")
            logger.info("jwks.loaded", extra={"active_kid": active.kid, "published": len(rows)})
            return cls(KeyPair(active.kid, "", {}), [{"kid": row.kid} for row in rows], AuthCache())

        def active(self) -> KeyPair:
            return self._active

        def public_key(self, kid: str) -> Any:
            for jwk in self._published:
                if jwk["kid"] == kid:
                    return jwk
            raise KeyError(f"unknown kid {kid}")

        def jwks(self) -> dict[str, Any]:
            return {"keys": self._published}

        def store_refresh_token(self, user_id: str, token_hash: str, ttl_seconds: int) -> None:
            self._cache.set(f"refresh:{token_hash}", {"user_id": user_id})

        def consume_refresh_token(self, token_hash: str) -> dict[str, Any] | None:
            record = self._cache.get(f"refresh:{token_hash}")
            if record is not None:
                self._cache.invalidate(f"refresh:{token_hash}")  # one-time use
            return record
''',
)

# ----------------------------------------------------------------------------- user-service

_module(
    "user-service",
    "profiles",
    '''
    """Customer profiles with a cache-aside Redis layer."""

    from __future__ import annotations

    from typing import Any

    from novacart_common.logging import get_logger
    from user_service.addresses import AddressBook
    from user_service.cache import UserCache
    from user_service.db.repository import UserRepository
    from user_service.events.producer import EventPublisher
    from user_service.events.schemas import UserUpdated

    logger = get_logger(__name__)

    DEFAULT_LOCALE = "en-US"
    MUTABLE_FIELDS = frozenset({"display_name", "locale"})


    class UserNotFound(LookupError):
        pass


    class ProfileService:
        def __init__(self, repository: UserRepository, cache: UserCache, events: EventPublisher) -> None:
            self._repository = repository
            self._cache = cache
            self._events = events
            self._addresses = AddressBook()

        @classmethod
        def from_settings(cls) -> ProfileService:
            return cls(UserRepository(), UserCache(), EventPublisher())

        async def get_user(self, user_id: str) -> dict[str, Any]:
            profile = self._cache.get(user_id)
            if profile is None:
                profile = self._repository.get(user_id)
                if profile is None:
                    raise UserNotFound(user_id)
                self._cache.set(user_id, profile)
            locale = profile.get("locale") or DEFAULT_LOCALE
            return {**profile, "locale": locale}

        async def update_user(self, user_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            changes = {k: v for k, v in (payload or {}).items() if k in MUTABLE_FIELDS}
            self._repository.bulk_update_status([user_id], "active")
            self._cache.invalidate(user_id)
            self._events.publish(UserUpdated(user_id=user_id, changed_fields=sorted(changes)))
            return await self.get_user(user_id)

        async def list_addresses(self, user_id: str, limit: int = 50) -> dict[str, Any]:
            return {"user_id": user_id, "addresses": self._addresses.list_for(user_id)[:limit]}

        async def add_address(self, user_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            return self._addresses.add(user_id, payload or {})
''',
    primary="ProfileService",
)

_module(
    "user-service",
    "addresses",
    '''
    """Address validation and normalisation."""

    from __future__ import annotations

    import re
    from typing import Any

    SUPPORTED_COUNTRIES = frozenset({"US", "CA", "GB", "DE", "FR", "NL", "IN", "AU"})
    POSTAL_CODE_PATTERNS = {
        "US": re.compile(r"^\\d{5}(-\\d{4})?$"),
        "GB": re.compile(r"^[A-Z]{1,2}\\d[A-Z\\d]? ?\\d[A-Z]{2}$"),
        "IN": re.compile(r"^\\d{6}$"),
    }


    class InvalidAddress(ValueError):
        pass


    class AddressBook:
        def __init__(self) -> None:
            self._by_user: dict[str, list[dict[str, Any]]] = {}

        def list_for(self, user_id: str) -> list[dict[str, Any]]:
            return list(self._by_user.get(user_id, []))

        def add(self, user_id: str, address: dict[str, Any]) -> dict[str, Any]:
            normalised = normalise(address)
            self._by_user.setdefault(user_id, []).append(normalised)
            return normalised


    def normalise(address: dict[str, Any]) -> dict[str, Any]:
        country = str(address.get("country", "")).upper()
        if country not in SUPPORTED_COUNTRIES:
            raise InvalidAddress(f"unsupported country: {country!r}")
        postal_code = str(address.get("postal_code", "")).strip().upper()
        pattern = POSTAL_CODE_PATTERNS.get(country)
        if pattern and not pattern.match(postal_code):
            raise InvalidAddress(f"invalid postal code for {country}")
        return {**address, "country": country, "postal_code": postal_code}
''',
)

# ----------------------------------------------------------------------------- product-service

_module(
    "product-service",
    "catalog",
    '''
    """Catalogue reads (cache-aside) and price updates."""

    from __future__ import annotations

    from typing import Any

    from novacart_common.logging import get_logger
    from product_service.cache import ProductCache
    from product_service.db.repository import ProductRepository
    from product_service.events.producer import EventPublisher
    from product_service.events.schemas import PriceChanged, ProductUpdated
    from product_service.pricing import PriceEngine

    logger = get_logger(__name__)


    class CatalogService:
        def __init__(self, repository: ProductRepository, cache: ProductCache, pricing: PriceEngine, events: EventPublisher) -> None:
            self._repository = repository
            self._cache = cache
            self._pricing = pricing
            self._events = events

        @classmethod
        def from_settings(cls) -> CatalogService:
            return cls(ProductRepository(), ProductCache(), PriceEngine(), EventPublisher())

        async def get_product(self, product_id: str) -> dict[str, Any]:
            product = self._cache.get(product_id)
            if product is None:
                product = self._repository.get(product_id) or {}
                self._cache.set(product_id, product)
            price = self._pricing.effective_price(product.get("list_price_minor", 0), product.get("promotions", []))
            if price < 0:
                raise ValueError("effective price must be non-negative")
            return {**product, "price_minor": price}

        async def list_products(self, limit: int = 50) -> dict[str, Any]:
            return {"items": self._repository.list_recent(limit)}

        async def update_price(self, product_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            request = payload or {}
            self._events.publish(
                PriceChanged(
                    sku=request["sku"],
                    old_price_minor=request["old_price_minor"],
                    new_price_minor=request["new_price_minor"],
                    currency=request.get("currency", "USD"),
                )
            )
            self._events.publish(ProductUpdated(product_id=product_id, sku_ids=[request["sku"]], status="active"))
            self._cache.invalidate(product_id)
            return await self.get_product(product_id)
''',
    primary="CatalogService",
)

_module(
    "product-service",
    "pricing",
    '''
    """Effective price calculation (list price minus stacked promotions)."""

    from __future__ import annotations

    from typing import Any

    MAX_STACKED_PROMOTIONS = 2


    class PriceEngine:
        def effective_price(self, list_price_minor: int, promotions: list[dict[str, Any]]) -> int:
            discount_minor = 0
            for promotion in sorted(promotions, key=lambda p: p.get("priority", 0))[:MAX_STACKED_PROMOTIONS]:
                if promotion["type"] == "percent":
                    discount_minor += list_price_minor * promotion["value"] // 100
                else:
                    discount_minor += promotion["value"]
            return max(list_price_minor - discount_minor, 0)
''',
)

# ----------------------------------------------------------------------------- inventory-service

_module(
    "inventory-service",
    "reservations",
    '''
    """Stock reservations. Rows are locked in SKU order to avoid deadlocks."""

    from __future__ import annotations

    from datetime import UTC, datetime, timedelta
    from typing import Any

    from sqlalchemy import select

    from inventory_service.cache import InventoryCache
    from inventory_service.db.database import session_scope
    from inventory_service.db.models import Reservation, StockLevel
    from inventory_service.events.producer import EventPublisher
    from inventory_service.events.schemas import InventoryReserved
    from novacart_common.ids import new_id
    from novacart_common.logging import get_logger

    logger = get_logger(__name__)

    RESERVATION_TTL_SECONDS = 900


    class InsufficientStock(Exception):
        pass


    class ReservationManager:
        def __init__(self, cache: InventoryCache, events: EventPublisher) -> None:
            self._cache = cache
            self._events = events

        @classmethod
        def from_settings(cls) -> ReservationManager:
            return cls(InventoryCache(), EventPublisher())

        async def get_stock(self, sku: str) -> dict[str, Any]:
            cached = self._cache.get(sku)
            if cached is not None:
                return cached
            with session_scope() as session:
                levels = session.scalars(select(StockLevel).where(StockLevel.sku == sku)).all()
                stock = {"sku": sku, "available": sum(level.on_hand - level.reserved for level in levels)}
            self._cache.set(sku, stock)
            return stock

        async def create_reservation(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            request = payload or {}
            lines = sorted(request["lines"], key=lambda line: line["sku"])
            expires_at = datetime.now(UTC) + timedelta(seconds=RESERVATION_TTL_SECONDS)
            with session_scope() as session:
                for line in lines:
                    level = session.scalars(
                        select(StockLevel).where(StockLevel.sku == line["sku"]).with_for_update()
                    ).first()
                    if level is None or level.on_hand - level.reserved < line["quantity"]:
                        raise InsufficientStock(line["sku"])
                    level.reserved += line["quantity"]
                    session.add(
                        Reservation(id=new_id(), order_id=request["order_id"], sku=line["sku"],
                                    quantity=line["quantity"], status="held", expires_at=expires_at)
                    )
            for line in lines:
                self._cache.invalidate(line["sku"])
            self._events.publish(InventoryReserved(reservation_id=request["order_id"], order_id=request["order_id"], sku_count=len(lines)))
            return {"order_id": request["order_id"], "expires_at": expires_at.isoformat()}

        async def release_reservation(self, reservation_id: str) -> dict[str, Any]:
            with session_scope() as session:
                reservation = session.get(Reservation, reservation_id, with_for_update=True)
                if reservation is None or reservation.status != "held":
                    return {"released": False}
                level = session.scalars(
                    select(StockLevel).where(StockLevel.sku == reservation.sku).with_for_update()
                ).first()
                level.reserved -= reservation.quantity
                reservation.status = "released"
            return {"released": True}
''',
    primary="ReservationManager",
)

_module(
    "inventory-service",
    "warehouse_sync",
    '''
    """Applies stock deltas from the StockSync WMS every 60 seconds."""

    from __future__ import annotations

    from typing import Any

    from sqlalchemy import select

    from inventory_service.clients.stocksync_client import StockSyncClient
    from inventory_service.db.database import session_scope
    from inventory_service.db.models import StockLevel
    from inventory_service.events.producer import EventPublisher
    from inventory_service.events.schemas import InventoryDepleted
    from novacart_common.logging import get_logger

    logger = get_logger(__name__)

    SYNC_INTERVAL_SECONDS = 60
    RECONCILIATION_TOLERANCE_UNITS = 2


    class WarehouseSyncJob:
        def __init__(self, wms: StockSyncClient, events: EventPublisher) -> None:
            self._wms = wms
            self._events = events
            self._last_sequence: dict[str, int] = {}

        async def run_once(self) -> int:
            deltas = (await self._wms.call("deltas", since=min(self._last_sequence.values(), default=0)))["deltas"]
            applied = 0
            with session_scope() as session:
                for delta in sorted(deltas, key=lambda d: d["sequence"]):
                    if delta["sequence"] <= self._last_sequence.get(delta["warehouse_id"], 0):
                        continue
                    level = session.scalars(
                        select(StockLevel)
                        .where(StockLevel.sku == delta["sku"], StockLevel.warehouse_id == delta["warehouse_id"])
                        .with_for_update()
                    ).first()
                    if level is None:
                        continue
                    level.on_hand += delta["quantity_change"]
                    self._last_sequence[delta["warehouse_id"]] = delta["sequence"]
                    applied += 1
                    if level.on_hand - level.reserved <= 0:
                        self._events.publish(InventoryDepleted(sku=delta["sku"], warehouse_id=delta["warehouse_id"]))
            logger.info("wms.sync_applied", extra={"deltas": len(deltas), "applied": applied})
            return applied

        def reconcile(self, wms_levels: dict[str, int], local_levels: dict[str, int]) -> list[dict[str, Any]]:
            mismatches = []
            for sku, wms_on_hand in wms_levels.items():
                drift = local_levels.get(sku, 0) - wms_on_hand
                if abs(drift) > RECONCILIATION_TOLERANCE_UNITS:
                    mismatches.append({"sku": sku, "drift": drift})
            if mismatches:
                logger.error("reconciliation.mismatch", extra={"skus": len(mismatches)})
            return mismatches
''',
)

# ----------------------------------------------------------------------------- cart-service

_module(
    "cart-service",
    "cart_store",
    '''
    """Carts live in Redis as one hash per cart, expiring after 7 days of inactivity."""

    from __future__ import annotations

    import json
    from typing import Any

    import redis

    from cart_service.cart_pricing import CartPricer
    from cart_service.config import settings
    from novacart_common.logging import get_logger

    logger = get_logger(__name__)


    class CartStore:
        def __init__(self, client: redis.Redis | None = None) -> None:
            self._redis = client or redis.Redis.from_url(settings.redis_url, socket_timeout=0.2)

        def _key(self, cart_id: str) -> str:
            return f"cart:{cart_id}"

        def lines(self, cart_id: str) -> dict[str, int]:
            raw = self._redis.hgetall(self._key(cart_id))
            return {sku.decode(): int(quantity) for sku, quantity in raw.items()}

        def add(self, cart_id: str, sku: str, quantity: int) -> None:
            pipe = self._redis.pipeline()
            pipe.hincrby(self._key(cart_id), sku, quantity)
            pipe.expire(self._key(cart_id), settings.cache_ttl_seconds)
            pipe.execute()

        def remove(self, cart_id: str, sku: str) -> None:
            self._redis.hdel(self._key(cart_id), sku)

        def freeze(self, cart_id: str) -> None:
            self._redis.rename(self._key(cart_id), f"cart:checked_out:{cart_id}")


    class CartService:
        def __init__(self, store: CartStore, pricer: CartPricer) -> None:
            self._store = store
            self._pricer = pricer

        @classmethod
        def from_settings(cls) -> CartService:
            return cls(CartStore(), CartPricer.from_settings())

        async def get_cart(self, cart_id: str) -> dict[str, Any]:
            return await self._pricer.price(cart_id, self._store.lines(cart_id))

        async def add_item(self, cart_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            request = payload or {}
            self._store.add(cart_id, request["sku"], int(request.get("quantity", 1)))
            return await self.get_cart(cart_id)

        async def remove_item(self, cart_id: str, sku: str) -> dict[str, Any]:
            self._store.remove(cart_id, sku)
            return await self.get_cart(cart_id)

        async def checkout_cart(self, cart_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            priced = await self.get_cart(cart_id)
            self._store.freeze(cart_id)
            logger.info("cart.checked_out", extra={"cart_id": cart_id, "total_minor": priced["total_minor"]})
            return {**priced, "checkout_payload": json.dumps(payload or {})}
''',
    primary="CartService",
)

_module(
    "cart-service",
    "cart_pricing",
    '''
    """Prices cart lines with current SKU prices from product-service."""

    from __future__ import annotations

    from typing import Any

    from cart_service.clients.inventory_client import InventoryClient
    from cart_service.clients.product_client import ProductClient
    from novacart_common.http import UpstreamUnavailable
    from novacart_common.logging import get_logger

    logger = get_logger(__name__)


    class CartPricer:
        def __init__(self, products: ProductClient, inventory: InventoryClient) -> None:
            self._products = products
            self._inventory = inventory

        @classmethod
        def from_settings(cls) -> CartPricer:
            return cls(ProductClient(), InventoryClient())

        async def price(self, cart_id: str, lines: dict[str, int]) -> dict[str, Any]:
            prices = await self._fetch_prices(list(lines))
            priced_lines, total_minor = [], 0
            for sku, quantity in lines.items():
                unit_price = prices.get(sku)
                if unit_price is None:
                    priced_lines.append({"sku": sku, "quantity": quantity, "available": False})
                    continue
                total_minor += unit_price * quantity
                priced_lines.append({"sku": sku, "quantity": quantity, "unit_price_minor": unit_price})
            return {"cart_id": cart_id, "lines": priced_lines, "total_minor": total_minor}

        async def _fetch_prices(self, skus: list[str]) -> dict[str, int]:
            prices: dict[str, int] = {}
            for sku in skus:
                try:
                    product = await self._products.get_product(sku)
                except UpstreamUnavailable:
                    logger.warning("cart.price_unavailable", extra={"sku": sku})
                    continue
                if product.get("status") == "active":
                    prices[sku] = product["price_minor"]
            return prices
''',
)

# ----------------------------------------------------------------------------- order-service

_module(
    "order-service",
    "checkout",
    '''
    """Checkout orchestration: reserve stock, authorise payment, record the order.

    On payment failure the inventory reservation is released (compensation).
    """

    from __future__ import annotations

    from typing import Any

    from novacart_common.http import UpstreamUnavailable
    from novacart_common.logging import get_logger
    from order_service.clients.cart_client import CartClient
    from order_service.clients.inventory_client import InventoryClient
    from order_service.clients.payment_client import PaymentClient
    from order_service.clients.user_client import UserClient
    from order_service.db.repository import OrderRepository
    from order_service.events.producer import EventPublisher
    from order_service.events.schemas import OrderCancelled, OrderCreated

    logger = get_logger(__name__)


    class CheckoutFailed(Exception):
        pass


    class CheckoutOrchestrator:
        def __init__(self, carts: CartClient, inventory: InventoryClient, payments: PaymentClient,
                     users: UserClient, orders: OrderRepository, events: EventPublisher) -> None:
            self._carts = carts
            self._inventory = inventory
            self._payments = payments
            self._users = users
            self._orders = orders
            self._events = events

        @classmethod
        def from_settings(cls) -> CheckoutOrchestrator:
            return cls(CartClient(), InventoryClient(), PaymentClient(), UserClient(), OrderRepository(), EventPublisher())

        async def create_order(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            request = payload or {}
            cart = await self._carts.get_cart(request["cart_id"])
            shipping_address = request.get("shipping_address") or await self._default_address(request["user_id"])
            order = self._orders.create(user_id=request["user_id"], cart_id=cart["cart_id"],
                                        total_minor=cart["total_minor"], currency="USD", status="pending", payment_id="")
            await self._inventory.reserve(order["id"], [{"sku": l["sku"], "quantity": l["quantity"]} for l in cart["lines"]])
            try:
                payment = await self._payments.authorize(order["id"], str(cart["total_minor"] / 100), "USD", request["idempotency_key"])
            except UpstreamUnavailable as exc:
                logger.error("checkout.payment_failed", extra={"order_id": order["id"], "error": str(exc)})
                self._orders.bulk_update_status([order["id"]], "payment_failed")
                raise CheckoutFailed("payment unavailable") from exc
            self._orders.bulk_update_status([order["id"]], "confirmed")
            self._events.publish(OrderCreated(order_id=order["id"], user_id=request["user_id"],
                                              total_minor=cart["total_minor"], currency="USD"))
            return {**order, "status": "confirmed", "payment_id": payment["id"], "shipping_address": shipping_address}

        async def get_order(self, order_id: str) -> dict[str, Any]:
            return self._orders.get(order_id) or {}

        async def cancel_order(self, order_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            self._orders.bulk_update_status([order_id], "cancelled")
            self._events.publish(OrderCancelled(order_id=order_id, reason=(payload or {}).get("reason", "customer_request")))
            return {"order_id": order_id, "status": "cancelled"}

        async def _default_address(self, user_id: str) -> dict[str, Any] | None:
            try:
                user = await self._users.get_user(user_id)
            except UpstreamUnavailable:
                return None  # soft dependency: address can be collected later
            addresses = user.get("addresses", [])
            return addresses[0] if addresses else None
''',
    primary="CheckoutOrchestrator",
)

_module(
    "order-service",
    "order_saga",
    '''
    """Order saga: advances order state from payment and inventory events."""

    from __future__ import annotations

    from typing import Any

    from novacart_common.logging import get_logger
    from order_service.db.repository import OrderRepository

    logger = get_logger(__name__)

    TRANSITIONS: dict[str, set[str]] = {
        "pending": {"confirmed", "payment_failed", "cancelled"},
        "confirmed": {"paid", "cancelled"},
        "paid": {"fulfilling", "refunded"},
        "fulfilling": {"shipped"},
    }


    class InvalidTransition(Exception):
        pass


    class OrderSaga:
        def __init__(self, orders: OrderRepository) -> None:
            self._orders = orders

        def _advance(self, order_id: str, target: str) -> None:
            order = self._orders.get(order_id)
            if order is None:
                logger.warning("saga.unknown_order", extra={"order_id": order_id})
                return
            if target not in TRANSITIONS.get(order["status"], set()):
                raise InvalidTransition(f"{order['status']} -> {target}")
            self._orders.bulk_update_status([order_id], target)

        async def on_payment_completed(self, event: dict[str, Any]) -> None:
            self._advance(event["order_id"], "paid")

        async def on_payment_failed(self, event: dict[str, Any]) -> None:
            self._advance(event["order_id"], "payment_failed")

        async def on_inventory_reserved(self, event: dict[str, Any]) -> None:
            logger.info("saga.inventory_reserved", extra={"order_id": event["order_id"]})
''',
)

# ----------------------------------------------------------------------------- payment-service

_module(
    "payment-service",
    "processor",
    '''
    """Payment authorisation and capture through PayFlux, with RiskShield scoring."""

    from __future__ import annotations

    from typing import Any

    from novacart_common.logging import get_logger
    from payment_service.cache import PaymentCache
    from payment_service.clients.payflux_client import PayFluxClient, ProviderUnavailable
    from payment_service.clients.riskshield_client import RiskShieldClient
    from payment_service.db.repository import PaymentRepository
    from payment_service.events.producer import EventPublisher
    from payment_service.events.schemas import PaymentCompleted, PaymentFailed
    from payment_service.idempotency import IdempotencyStore
    from payment_service.refunds import RefundService

    logger = get_logger(__name__)

    ZERO_DECIMAL_CURRENCIES = frozenset({"JPY", "KRW"})
    RISK_DECLINE_THRESHOLD = 0.85


    class PaymentDeclined(Exception):
        def __init__(self, decline_code: str) -> None:
            super().__init__(decline_code)
            self.decline_code = decline_code


    def to_minor_units(amount: str, currency: str) -> int:
        if currency in ZERO_DECIMAL_CURRENCIES:
            return int(amount)
        return int(round(float(amount) * 100))


    class PaymentProcessor:
        def __init__(self, repository: PaymentRepository, psp: PayFluxClient, risk: RiskShieldClient,
                     idempotency: IdempotencyStore, refunds: RefundService, events: EventPublisher) -> None:
            self._repository = repository
            self._psp = psp
            self._risk = risk
            self._idempotency = idempotency
            self._refunds = refunds
            self._events = events

        @classmethod
        def from_settings(cls) -> PaymentProcessor:
            repository = PaymentRepository()
            return cls(repository, PayFluxClient(), RiskShieldClient(), IdempotencyStore(PaymentCache()),
                       RefundService(repository), EventPublisher())

        async def create_payment(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            request = payload or {}
            key = request["idempotency_key"]
            previous = self._idempotency.lookup(key)
            if previous is not None:
                return previous
            amount_minor = to_minor_units(request["amount"], request["currency"])
            score = (await self._risk.call("score", order_id=request["order_id"], amount_minor=amount_minor))["score"]
            if score >= RISK_DECLINE_THRESHOLD:
                self._events.publish(PaymentFailed(payment_id="", order_id=request["order_id"], decline_code="risk_declined"))
                raise PaymentDeclined("risk_declined")
            card_token = request["card_token"]
            try:
                authorization = await self._psp.call(
                    "authorizations", amount_minor=amount_minor, currency=request["currency"],
                    token=card_token, idempotency_key=key,
                )
            except ProviderUnavailable:
                logger.warning("payment.provider_unavailable", extra={"order_id": request["order_id"]})
                raise
            payment = self._repository.create(
                order_id=request["order_id"], amount_minor=amount_minor, currency=request["currency"],
                status="authorized", psp_reference=authorization["id"], idempotency_key=key,
            )
            self._idempotency.complete(key, payment)
            return payment

        async def capture_payment(self, payment_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            payment = self._repository.get(payment_id)
            if payment is None or payment["status"] != "authorized":
                raise PaymentDeclined("not_capturable")
            await self._psp.call("captures", authorization_id=payment["psp_reference"])
            self._repository.bulk_update_status([payment_id], "captured")
            self._events.publish(PaymentCompleted(payment_id=payment_id, order_id=payment["order_id"],
                                                  amount_minor=payment["amount_minor"], currency=payment["currency"]))
            return {**payment, "status": "captured"}

        async def get_payment(self, payment_id: str) -> dict[str, Any]:
            return self._repository.get(payment_id) or {}

        async def create_refund(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            return await self._refunds.issue(payload or {})
''',
    primary="PaymentProcessor",
)

_module(
    "payment-service",
    "idempotency",
    '''
    """Idempotency keys: a retried request returns the original response instead of charging twice."""

    from __future__ import annotations

    from typing import Any

    from novacart_common.logging import get_logger
    from payment_service.cache import PaymentCache

    logger = get_logger(__name__)


    class IdempotencyStore:
        def __init__(self, cache: PaymentCache) -> None:
            self._cache = cache

        def lookup(self, key: str) -> dict[str, Any] | None:
            response = self._cache.get(f"idem:{key}")
            if response is not None:
                logger.info("idempotency.replay", extra={"key": key})
            return response

        def complete(self, key: str, response: dict[str, Any]) -> None:
            self._cache.set(f"idem:{key}", response)
''',
)

_module(
    "payment-service",
    "refunds",
    '''
    """Refunds against captured payments."""

    from __future__ import annotations

    from typing import Any

    from novacart_common.logging import get_logger
    from payment_service.db.repository import PaymentRepository

    logger = get_logger(__name__)


    class RefundRejected(Exception):
        pass


    class RefundService:
        def __init__(self, payments: PaymentRepository) -> None:
            self._payments = payments

        async def issue(self, request: dict[str, Any]) -> dict[str, Any]:
            payment = self._payments.get(request["payment_id"])
            if payment is None or payment["status"] != "captured":
                raise RefundRejected("payment is not captured")
            amount_minor = int(request.get("amount_minor", payment["amount_minor"]))
            if amount_minor > payment["amount_minor"]:
                raise RefundRejected("refund exceeds captured amount")
            logger.info("refund.requested", extra={"payment_id": payment["id"], "amount_minor": amount_minor})
            return {"payment_id": payment["id"], "amount_minor": amount_minor, "status": "pending"}
''',
)

# ----------------------------------------------------------------------------- notification-service

_module(
    "notification-service",
    "dispatcher",
    '''
    """Delivers notifications through MailRelay (email) and TextBridge (SMS)."""

    from __future__ import annotations

    import asyncio
    from typing import Any

    from notification_service.clients.mailrelay_client import MailRelayClient
    from notification_service.clients.textbridge_client import TextBridgeClient
    from notification_service.db.repository import NotificationRepository
    from notification_service.template_renderer import TemplateRenderer
    from novacart_common.http import UpstreamUnavailable
    from novacart_common.logging import get_logger

    logger = get_logger(__name__)

    MAX_ATTEMPTS = 5
    BASE_BACKOFF_SECONDS = 2.0


    class NotificationDispatcher:
        def __init__(self, email: MailRelayClient, sms: TextBridgeClient,
                     repository: NotificationRepository, renderer: TemplateRenderer) -> None:
            self._channels = {"email": email, "sms": sms}
            self._repository = repository
            self._renderer = renderer

        @classmethod
        def from_settings(cls) -> NotificationDispatcher:
            return cls(MailRelayClient(), TextBridgeClient(), NotificationRepository(), TemplateRenderer())

        async def send_notification(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
            request = payload or {}
            body = self._renderer.render(request["template"], request.get("variables", {}))
            record = self._repository.create(user_id=request["user_id"], channel=request["channel"],
                                             template=request["template"], status="queued")
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    await self._channels[request["channel"]].call("messages", to=request["to"], body=body)
                    self._repository.bulk_update_status([record["id"]], "sent")
                    return {**record, "status": "sent", "attempts": attempt}
                except UpstreamUnavailable as exc:
                    logger.warning("notification.delivery_retry", extra={"attempt": attempt, "error": str(exc)})
                    await asyncio.sleep(BASE_BACKOFF_SECONDS * 2 ** (attempt - 1))
            self._repository.bulk_update_status([record["id"]], "failed")
            return {**record, "status": "failed", "attempts": MAX_ATTEMPTS}

        async def get_notification(self, notification_id: str) -> dict[str, Any]:
            return self._repository.get(notification_id) or {}
''',
    primary="NotificationDispatcher",
)

_module(
    "notification-service",
    "template_renderer",
    '''
    """Renders notification templates; unknown templates fall back to a generic update."""

    from __future__ import annotations

    import string
    from typing import Any

    FALLBACK_TEMPLATE = "generic_update"
    TEMPLATES = {
        "order_confirmation": "Thanks for your order $order_id. Total: $total.",
        "payment_failed": "We could not process the payment for order $order_id.",
        "refund_issued": "Your refund of $amount is on its way.",
        FALLBACK_TEMPLATE: "There is an update on your NovaCart order $order_id.",
    }


    class TemplateRenderer:
        def __init__(self, templates: dict[str, str] | None = None) -> None:
            self._templates = templates or TEMPLATES

        def render(self, name: str, variables: dict[str, Any]) -> str:
            template = self._templates.get(name) or self._templates[FALLBACK_TEMPLATE]
            return string.Template(template).safe_substitute(variables)
''',
)

# ----------------------------------------------------------------------------- recommendation-service

_module(
    "recommendation-service",
    "ranker",
    '''
    """Scores candidate products; falls back to popular items when features are missing."""

    from __future__ import annotations

    from typing import Any

    from novacart_common.logging import get_logger
    from recommendation_service.feature_store import FeatureStore, FeatureStoreUnavailable

    logger = get_logger(__name__)

    DEFAULT_USER_FEATURES = {"affinity": {}, "segment": "new_visitor"}
    MAX_RESULTS = 20


    class Ranker:
        def __init__(self, features: FeatureStore) -> None:
            self._features = features

        @classmethod
        def from_settings(cls) -> Ranker:
            return cls(FeatureStore())

        async def recommend_for_user(self, user_id: str) -> dict[str, Any]:
            try:
                features = self._features.user_features(user_id) or DEFAULT_USER_FEATURES
                candidates = self._features.candidates(features["segment"])
            except FeatureStoreUnavailable:
                logger.warning("ranker.fallback_popular", extra={"user_id": user_id})
                return {"user_id": user_id, "items": self._features.popular_items()[:MAX_RESULTS], "fallback": True}
            scored = sorted(
                candidates, key=lambda item: features["affinity"].get(item["category"], 0.0) + item["popularity"], reverse=True
            )
            return {"user_id": user_id, "items": scored[:MAX_RESULTS], "fallback": False}

        async def similar_products(self, product_id: str) -> dict[str, Any]:
            return {"product_id": product_id, "items": self._features.neighbours(product_id)[:MAX_RESULTS]}
''',
    primary="Ranker",
)

_module(
    "recommendation-service",
    "feature_store",
    '''
    """Reads precomputed features and candidate lists from Redis."""

    from __future__ import annotations

    from typing import Any

    import redis

    from novacart_common.logging import get_logger
    from recommendation_service.cache import RecommendationCache

    logger = get_logger(__name__)


    class FeatureStoreUnavailable(RuntimeError):
        pass


    class FeatureStore:
        def __init__(self, cache: RecommendationCache | None = None) -> None:
            self._cache = cache or RecommendationCache()

        def _read(self, key: str) -> Any:
            try:
                return self._cache.get(key)
            except redis.RedisError as exc:
                raise FeatureStoreUnavailable(str(exc)) from exc

        def user_features(self, user_id: str) -> dict[str, Any] | None:
            return self._read(f"user:{user_id}")

        def candidates(self, segment: str) -> list[dict[str, Any]]:
            return (self._read(f"candidates:{segment}") or {}).get("items", [])

        def neighbours(self, product_id: str) -> list[dict[str, Any]]:
            return (self._read(f"similar:{product_id}") or {}).get("items", [])

        def popular_items(self) -> list[dict[str, Any]]:
            return (self._read("popular:global") or {}).get("items", [])
''',
)

# ----------------------------------------------------------------------------- search-service

_module(
    "search-service",
    "query_builder",
    '''
    """Builds OpenSearch queries from user input and serves cached results."""

    from __future__ import annotations

    import re
    from typing import Any

    from opensearchpy import OpenSearch

    from novacart_common.logging import get_logger
    from search_service.cache import SearchCache
    from search_service.config import settings

    logger = get_logger(__name__)

    INDEX = "products-v7"
    _RESERVED = re.compile(r'([+\\-=&|><!(){}\\[\\]^"~*?:\\\\/])')


    def escape_query(text: str) -> str:
        """Escape query_string syntax so user input is treated as plain text."""
        return _RESERVED.sub(r"\\\\\\1", text.strip())


    class SearchService:
        def __init__(self, client: OpenSearch, cache: SearchCache) -> None:
            self._client = client
            self._cache = cache

        @classmethod
        def from_settings(cls) -> SearchService:
            return cls(OpenSearch(settings.opensearch_url, timeout=settings.http_timeout_seconds), SearchCache())

        def build(self, q: str, limit: int) -> dict[str, Any]:
            terms = escape_query(q)
            return {
                "size": limit,
                "query": {
                    "bool": {
                        "must": [{"query_string": {"query": terms, "fields": ["title^3", "brand^2", "description"]}}],
                        "filter": [{"term": {"status": "active"}}],
                    }
                },
            }

        async def search_products(self, q: str, limit: int = 20) -> dict[str, Any]:
            cache_key = f"q:{q.lower()}:{limit}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
            response = self._client.search(index=INDEX, body=self.build(q, limit))
            result = {"total": response["hits"]["total"]["value"], "items": [h["_source"] for h in response["hits"]["hits"]]}
            self._cache.set(cache_key, result)
            return result

        async def suggest(self, q: str, limit: int = 20) -> dict[str, Any]:
            body = {"suggest": {"titles": {"prefix": q, "completion": {"field": "title_suggest", "size": limit}}}}
            response = self._client.search(index=INDEX, body=body)
            return {"suggestions": [o["text"] for o in response["suggest"]["titles"][0]["options"]]}
''',
    primary="SearchService",
)

_module(
    "search-service",
    "indexer",
    '''
    """Keeps the products index in sync with catalogue events."""

    from __future__ import annotations

    from typing import Any

    from opensearchpy import OpenSearch, helpers

    from novacart_common.logging import get_logger
    from search_service.config import settings

    logger = get_logger(__name__)

    INDEX = "products-v7"
    BULK_SIZE = 500


    class ProductIndexer:
        def __init__(self, client: OpenSearch | None = None) -> None:
            self._client = client or OpenSearch(settings.opensearch_url, timeout=10)
            self._pending: list[dict[str, Any]] = []

        def on_product_updated(self, event: dict[str, Any]) -> None:
            self._pending.append({"_op_type": "update", "_index": INDEX, "_id": event["product_id"],
                                  "doc": {"status": event["status"]}, "doc_as_upsert": True})
            if len(self._pending) >= BULK_SIZE:
                self.flush()

        def on_price_changed(self, event: dict[str, Any]) -> None:
            self._pending.append({"_op_type": "update", "_index": INDEX, "_id": event["sku"],
                                  "doc": {"price_minor": event["new_price_minor"]}})
            if len(self._pending) >= BULK_SIZE:
                self.flush()

        def flush(self) -> None:
            if not self._pending:
                return
            success, errors = helpers.bulk(self._client, self._pending, raise_on_error=False)
            if errors:
                logger.error("indexer.bulk_errors", extra={"failed": len(errors), "succeeded": success})
            self._pending.clear()
''',
)


# ============================================================================ faults

REGRESSIONS: dict[str, CodeFault] = {
    fault.service_id: fault
    for fault in (
        CodeFault(
            "api-gateway",
            "routing",
            (
                (
                    'url = f"{upstream.base_url}/v1/{service}/{path}"',
                    'url = f"{upstream.base_url}/{path}"',
                ),
            ),
            "Simplify upstream URL construction in the router",
            "Upstreams mount their routers at the root, so the gateway no longer needs to rebuild the /v1/<service> prefix.",
            "The router dropped the /v1/<service> prefix when building upstream URLs, so every proxied request hit a path the backends do not serve.",
            "Restore /v1/<service> prefix in upstream URLs",
            "clients received HTTP 502 for most proxied API calls; backends logged a flood of 404s",
            "httpx.HTTPStatusError",
            "upstream returned 404 for proxied path",
            "GET /v1/{service}/{path}",
            502,
        ),
        CodeFault(
            "auth-service",
            "tokens",
            (
                (
                    'algorithm="RS256", headers={"kid": key.kid}',
                    'algorithm="ES256", headers={"kid": key.kid}',
                ),
            ),
            "Prepare token signing for ES256 keys",
            "First step of the ECDSA migration; the key ring already serves both key types.",
            "Tokens were signed with ES256 while the active key in the ring was still RSA, so every token issuance raised InvalidKeyError.",
            "Sign tokens with the algorithm of the active key (revert to RS256)",
            "logins and token refreshes failed with HTTP 500; existing sessions kept working until their tokens expired",
            "jwt.exceptions.InvalidKeyError",
            "Expecting an EllipticCurvePrivateKey; got RSAPrivateKey",
            "POST /v1/auth/token",
            500,
        ),
        CodeFault(
            "user-service",
            "profiles",
            (
                (
                    'locale = profile.get("locale") or DEFAULT_LOCALE',
                    'locale = profile["preferences"]["locale"]',
                ),
            ),
            "Read locale from the nested preferences document",
            "Profiles now store locale under preferences; this aligns reads with the new write path.",
            "Profiles created before the preferences migration have no 'preferences' key, so reading them raised KeyError.",
            "Fall back to top-level locale when preferences are absent",
            "GET /v1/users/{user_id} returned HTTP 500 for older accounts; account pages and checkout address lookup failed",
            "KeyError",
            "KeyError: 'preferences'",
            "GET /v1/users/{user_id}",
            500,
        ),
        CodeFault(
            "product-service",
            "pricing",
            (
                (
                    "return max(list_price_minor - discount_minor, 0)",
                    "return list_price_minor - discount_minor",
                ),
            ),
            "Simplify effective price calculation",
            "The clamp is redundant because promotions are validated on creation.",
            "Stacked fixed-amount promotions can exceed the list price; without the clamp the effective price went negative and get_product raised ValueError.",
            "Clamp effective price at zero",
            "product pages for discounted items returned HTTP 500; cart pricing marked those items unavailable",
            "ValueError",
            "effective price must be non-negative",
            "GET /v1/products/{product_id}",
            500,
        ),
        CodeFault(
            "inventory-service",
            "warehouse_sync",
            (
                (
                    '                if delta["sequence"] <= self._last_sequence.get(delta["warehouse_id"], 0):\n                    continue\n',
                    "",
                ),
            ),
            "Apply every WMS delta in the sync batch",
            "Sequence numbers are per shard in StockSync, so the dedupe check was skipping valid deltas.",
            "Without the sequence check, deltas redelivered by StockSync were applied twice and local stock drifted from the WMS.",
            "Restore per-warehouse sequence dedupe for WMS deltas",
            "stock levels diverged from the warehouse system; some SKUs oversold while others showed out of stock",
            "ReconciliationMismatch",
            "reconciliation.mismatch: local stock drifted from WMS",
            "POST /v1/reservations",
            409,
        ),
        CodeFault(
            "cart-service",
            "cart_pricing",
            (("unit_price = prices.get(sku)", "unit_price = prices[sku]"),),
            "Tighten cart pricing lookups",
            "Every SKU in a cart should have a price; fail loudly instead of hiding data problems.",
            "Carts containing a discontinued or temporarily unpriced SKU raised KeyError instead of marking the line unavailable.",
            "Mark unpriced cart lines unavailable instead of raising",
            "GET /v1/carts/{cart_id} returned HTTP 500 for carts containing discontinued items",
            "KeyError",
            "KeyError raised while pricing cart line",
            "GET /v1/carts/{cart_id}",
            500,
        ),
        CodeFault(
            "order-service",
            "checkout",
            (
                (
                    'shipping_address = request.get("shipping_address") or await self._default_address(request["user_id"])',
                    'shipping_address = (await self._users.get_user(request["user_id"]))["addresses"][0]',
                ),
            ),
            "Always use the customer's default address at checkout",
            "Clients stopped sending shipping_address in the new checkout flow.",
            "Customers without a saved address hit IndexError, and user-service became a hard dependency of checkout.",
            "Restore optional shipping address handling in checkout",
            "POST /v1/orders failed with HTTP 500 for guest-converted and new customers",
            "IndexError",
            "IndexError: list index out of range",
            "POST /v1/orders",
            500,
        ),
        CodeFault(
            "payment-service",
            "processor",
            (
                (
                    'card_token = request["card_token"]',
                    'card_token = request["payment_method"]["token"]',
                ),
            ),
            "Support wallet payment methods in create_payment",
            "Reads the token from the new payment_method object used by wallet payments.",
            "Clients still sending the legacy card_token field caused KeyError before authorisation.",
            "Accept both card_token and payment_method.token",
            "POST /v1/payments returned HTTP 500 for card payments from older app versions",
            "KeyError",
            "KeyError: 'payment_method'",
            "POST /v1/payments",
            500,
        ),
        CodeFault(
            "notification-service",
            "template_renderer",
            (
                (
                    "template = self._templates.get(name) or self._templates[FALLBACK_TEMPLATE]",
                    "template = self._templates[name]",
                ),
            ),
            "Remove implicit template fallback",
            "Missing templates should surface as errors rather than silently sending a generic message.",
            "Events with templates not yet registered (refund_partial) raised KeyError, stalling the consumer batch.",
            "Restore fallback template for unknown notification types",
            "order and refund emails were delayed; the consumer retried failing batches",
            "KeyError",
            "KeyError: 'refund_partial'",
            "POST /v1/notifications",
            500,
        ),
        CodeFault(
            "recommendation-service",
            "ranker",
            (
                (
                    "features = self._features.user_features(user_id) or DEFAULT_USER_FEATURES",
                    "features = self._features.user_features(user_id)",
                ),
            ),
            "Drop default feature vector for unknown users",
            "Default features skewed offline evaluation; unknown users should get popular items instead.",
            "New visitors have no feature vector; indexing None raised TypeError instead of falling back to popular items.",
            "Restore default features for users without a feature vector",
            "GET /v1/recommendations/{user_id} returned HTTP 500 for new visitors; home page carousels were empty",
            "TypeError",
            "TypeError: 'NoneType' object is not subscriptable",
            "GET /v1/recommendations/{user_id}",
            500,
        ),
        CodeFault(
            "search-service",
            "query_builder",
            (("terms = escape_query(q)", "terms = q"),),
            "Allow advanced query syntax in product search",
            "Power users asked for quoted phrases and field filters.",
            "Unescaped user input reached OpenSearch query_string; queries with characters like '(' or '/' failed to parse.",
            "Escape user input before building query_string queries",
            "searches containing punctuation (e.g. 'usb-c (2m)') returned HTTP 500",
            "opensearchpy.exceptions.RequestError",
            "search_phase_execution_exception: Failed to parse query",
            "GET /v1/search",
            500,
        ),
    )
}

AUTH_FAULTS: dict[str, CodeFault] = {
    "api-gateway": CodeFault(
        "api-gateway",
        "auth_middleware",
        (
            ("JWKS_CACHE_TTL_SECONDS = 300", "JWKS_CACHE_TTL_SECONDS = 86400"),
            ("if expired or kid not in self._keys:", "if expired:"),
        ),
        "Reduce JWKS fetches from the gateway",
        "The gateway fetched JWKS thousands of times per minute; cache it for a day instead.",
        "After the weekly signing-key rotation, tokens carried a new kid that the gateway's day-long JWKS cache did not contain, and unknown kids no longer triggered a refresh.",
        "Refetch JWKS on unknown kid and restore 5-minute cache TTL",
        "newly issued tokens were rejected with HTTP 401 until gateway pods restarted",
        "AuthenticationError",
        "unknown signing key",
        "GET /v1/{service}/{path}",
        401,
    ),
    "auth-service": CodeFault(
        "auth-service",
        "tokens",
        (("CLOCK_SKEW_LEEWAY_SECONDS = 30", "CLOCK_SKEW_LEEWAY_SECONDS = 0"),),
        "Tighten token validation",
        "Security review recommended removing the leeway on iat/nbf checks.",
        "Pods whose clocks ran a few seconds behind rejected freshly issued tokens as 'not yet valid'.",
        "Restore 30s clock-skew leeway in token validation",
        "token introspection and refresh intermittently failed with HTTP 401",
        "jwt.exceptions.ImmatureSignatureError",
        "The token is not yet valid (iat)",
        "POST /v1/auth/introspect",
        401,
    ),
}

GATEWAY_RATE_LIMIT_FAULT = CodeFault(
    "api-gateway",
    "rate_limiter",
    (('key = f"ratelimit:{client_id}"', 'key = f"ratelimit:{client_ip}"'),),
    "Rate limit anonymous traffic by client IP",
    "Scrapers rotate client ids; keying by IP makes the limit harder to evade.",
    "Mobile users behind carrier-grade NAT share a handful of IPs, so thousands of customers drew from the same token bucket.",
    "Key rate limits by client id again",
    "mobile clients received HTTP 429 on most requests",
    "RateLimitExceeded",
    "rate limit exceeded",
    "GET /v1/{service}/{path}",
    429,
)
