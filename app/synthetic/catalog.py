"""NovaCart's organisation and architecture: the fixed facts every generated
artifact (code, config, docs, incidents, logs) is derived from."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.enums import AccessLevel, DependencyCriticality, ServiceTier
from app.synthetic.text import camel

REPOSITORY = "novacart"
INTERNAL_DOMAIN = "novacart.internal"
KAFKA_BOOTSTRAP = "kafka-1.novacart.internal:9092,kafka-2.novacart.internal:9092"


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str
    handler: str
    summary: str


@dataclass(frozen=True)
class HttpDependency:
    service: str
    criticality: DependencyCriticality
    purpose: str


@dataclass(frozen=True)
class ExternalProvider:
    name: str  # fictional vendor
    purpose: str
    client_module: str  # e.g. "payflux_client"
    base_url: str

    @property
    def client_class(self) -> str:
        return self.name.split()[0] + "Client"

    @property
    def settings_prefix(self) -> str:
        return self.client_module.removesuffix("_client")


@dataclass(frozen=True)
class ServiceConfig:
    """Production defaults; code, deploy manifests and docs all render these."""

    replicas: int
    cpu: str
    memory: str
    db_pool_size: int = 0
    db_max_overflow: int = 0
    db_pool_timeout_seconds: float = 3.0
    http_timeout_seconds: float = 2.0
    http_max_retries: int = 2
    cache_ttl_seconds: int = 300
    kafka_max_poll_records: int = 500
    consumer_concurrency: int = 8
    p99_latency_slo_ms: int = 300
    availability_slo: str = "99.9"


@dataclass(frozen=True)
class ServiceProfile:
    id: str
    display_name: str
    package: str
    team: str
    tier: ServiceTier
    description: str
    port: int
    start_version: tuple[int, int, int]
    config: ServiceConfig
    endpoints: tuple[Endpoint, ...]
    domain_modules: tuple[str, ...]
    postgres_db: str | None = None
    tables: tuple[str, ...] = ()
    redis_cluster: str | None = None
    redis_usage: str = ""
    produces: tuple[str, ...] = ()
    consumes: tuple[str, ...] = ()
    http_dependencies: tuple[HttpDependency, ...] = ()
    external: tuple[ExternalProvider, ...] = ()
    access_level: AccessLevel = AccessLevel.ENGINEERING  # for code and config
    extra_datastores: tuple[str, ...] = field(default=())

    @property
    def camel(self) -> str:
        return camel(self.id)

    @property
    def short(self) -> str:
        """ "payment-service" -> "Payment"; "api-gateway" -> "ApiGateway"."""
        return self.camel.removesuffix("Service")

    @property
    def env_prefix(self) -> str:
        return self.package.upper()

    @property
    def consumer_group(self) -> str | None:
        return f"{self.id}-consumer" if self.consumes else None

    @property
    def repo_path(self) -> str:
        return f"services/{self.id}"

    @property
    def oncall_channel(self) -> str:
        return f"#{self.team}-oncall"

    @property
    def datastores(self) -> list[str]:
        stores = []
        if self.postgres_db:
            stores.append(f"postgres:{self.postgres_db}")
        if self.redis_cluster:
            stores.append(f"redis:{self.redis_cluster}")
        if self.produces or self.consumes:
            stores.append("kafka")
        stores.extend(self.extra_datastores)
        return stores

    def has_http_dependency_on(self, service_id: str) -> bool:
        return any(dep.service == service_id for dep in self.http_dependencies)


HARD = DependencyCriticality.HARD
SOFT = DependencyCriticality.SOFT

SERVICES: tuple[ServiceProfile, ...] = (
    ServiceProfile(
        id="api-gateway",
        display_name="API Gateway",
        package="api_gateway",
        team="platform",
        tier=ServiceTier.TIER_0,
        description=(
            "Single public entry point for web and mobile clients. Terminates TLS, "
            "authenticates bearer tokens, applies per-client rate limits and routes "
            "requests to backend services."
        ),
        port=8080,
        start_version=(3, 4, 0),
        config=ServiceConfig(
            replicas=8,
            cpu="1000m",
            memory="768Mi",
            http_timeout_seconds=5.0,
            http_max_retries=1,
            cache_ttl_seconds=300,
            p99_latency_slo_ms=450,
            availability_slo="99.95",
        ),
        endpoints=(
            Endpoint(
                "GET",
                "/v1/{service}/{path}",
                "proxy_get",
                "Proxy read requests to the owning service",
            ),
            Endpoint(
                "POST",
                "/v1/{service}/{path}",
                "proxy_post",
                "Proxy write requests to the owning service",
            ),
        ),
        domain_modules=("routing", "rate_limiter", "auth_middleware"),
        redis_cluster="redis-gateway",
        redis_usage="token-bucket rate-limit counters and cached JWKS",
        http_dependencies=(
            HttpDependency("auth-service", HARD, "JWKS for bearer-token validation"),
            HttpDependency("user-service", HARD, "routes /v1/users/*"),
            HttpDependency("product-service", HARD, "routes /v1/products/*"),
            HttpDependency("cart-service", HARD, "routes /v1/carts/*"),
            HttpDependency("order-service", HARD, "routes /v1/orders/*"),
            HttpDependency("payment-service", HARD, "routes /v1/payments/*"),
            HttpDependency("search-service", SOFT, "routes /v1/search/*"),
            HttpDependency("recommendation-service", SOFT, "routes /v1/recommendations/*"),
        ),
    ),
    ServiceProfile(
        id="auth-service",
        display_name="Auth Service",
        package="auth_service",
        team="identity",
        tier=ServiceTier.TIER_0,
        description=(
            "Issues and refreshes OAuth2 access tokens (RS256 JWTs), publishes the JWKS "
            "used by the gateway to validate them, and rotates signing keys."
        ),
        port=8081,
        start_version=(1, 9, 0),
        config=ServiceConfig(
            replicas=6,
            cpu="750m",
            memory="512Mi",
            db_pool_size=15,
            db_max_overflow=5,
            db_pool_timeout_seconds=2.0,
            http_timeout_seconds=1.5,
            cache_ttl_seconds=900,
            p99_latency_slo_ms=200,
            availability_slo="99.95",
        ),
        endpoints=(
            Endpoint(
                "POST",
                "/v1/auth/token",
                "issue_token",
                "Exchange credentials for an access + refresh token",
            ),
            Endpoint("POST", "/v1/auth/refresh", "refresh_token", "Rotate a refresh token"),
            Endpoint(
                "POST", "/v1/auth/introspect", "introspect_token", "Validate a token server-side"
            ),
            Endpoint("GET", "/v1/auth/.well-known/jwks.json", "get_jwks", "Public signing keys"),
        ),
        domain_modules=("tokens", "jwks"),
        postgres_db="auth",
        tables=("credentials", "refresh_tokens", "signing_keys"),
        redis_cluster="redis-auth",
        redis_usage="refresh-token denylist and login throttling counters",
        produces=("user.logged_in",),
        http_dependencies=(HttpDependency("user-service", HARD, "loads user status during login"),),
        access_level=AccessLevel.SRE,
    ),
    ServiceProfile(
        id="user-service",
        display_name="User Service",
        package="user_service",
        team="identity",
        tier=ServiceTier.TIER_1,
        description="Owns customer profiles, addresses and preferences.",
        port=8082,
        start_version=(2, 1, 0),
        config=ServiceConfig(
            replicas=4,
            cpu="500m",
            memory="512Mi",
            db_pool_size=10,
            db_max_overflow=10,
            cache_ttl_seconds=600,
            p99_latency_slo_ms=250,
        ),
        endpoints=(
            Endpoint("GET", "/v1/users/{user_id}", "get_user", "Fetch a customer profile"),
            Endpoint("PATCH", "/v1/users/{user_id}", "update_user", "Update profile fields"),
            Endpoint(
                "GET", "/v1/users/{user_id}/addresses", "list_addresses", "List saved addresses"
            ),
            Endpoint("POST", "/v1/users/{user_id}/addresses", "add_address", "Save a new address"),
        ),
        domain_modules=("profiles", "addresses"),
        postgres_db="users",
        tables=("users", "addresses", "preferences"),
        redis_cluster="redis-users",
        redis_usage="cache-aside profile cache",
        produces=("user.updated",),
    ),
    ServiceProfile(
        id="product-service",
        display_name="Product Service",
        package="product_service",
        team="catalog",
        tier=ServiceTier.TIER_1,
        description="Product catalogue, SKUs and prices; source of truth for what is sold and for how much.",
        port=8083,
        start_version=(4, 2, 0),
        config=ServiceConfig(
            replicas=6,
            cpu="750m",
            memory="768Mi",
            db_pool_size=20,
            db_max_overflow=10,
            cache_ttl_seconds=300,
            p99_latency_slo_ms=200,
        ),
        endpoints=(
            Endpoint(
                "GET", "/v1/products/{product_id}", "get_product", "Fetch a product with its SKUs"
            ),
            Endpoint("GET", "/v1/products", "list_products", "Paginated catalogue listing"),
            Endpoint(
                "PUT",
                "/v1/products/{product_id}/price",
                "update_price",
                "Set the list price of a SKU",
            ),
        ),
        domain_modules=("catalog", "pricing"),
        postgres_db="catalog",
        tables=("products", "skus", "prices"),
        redis_cluster="redis-catalog",
        redis_usage="cache-aside product and price cache",
        produces=("product.updated", "price.changed"),
    ),
    ServiceProfile(
        id="inventory-service",
        display_name="Inventory Service",
        package="inventory_service",
        team="fulfillment",
        tier=ServiceTier.TIER_0,
        description=(
            "Tracks stock per SKU and warehouse, reserves stock for orders and "
            "synchronises levels with the StockSync warehouse management system."
        ),
        port=8084,
        start_version=(1, 14, 0),
        config=ServiceConfig(
            replicas=4,
            cpu="750m",
            memory="768Mi",
            db_pool_size=20,
            db_max_overflow=10,
            db_pool_timeout_seconds=3.0,
            cache_ttl_seconds=30,
            kafka_max_poll_records=250,
            consumer_concurrency=8,
            p99_latency_slo_ms=250,
            availability_slo="99.95",
        ),
        endpoints=(
            Endpoint("GET", "/v1/inventory/{sku}", "get_stock", "Available stock for a SKU"),
            Endpoint(
                "POST", "/v1/reservations", "create_reservation", "Reserve stock for an order"
            ),
            Endpoint(
                "DELETE",
                "/v1/reservations/{reservation_id}",
                "release_reservation",
                "Release a reservation",
            ),
        ),
        domain_modules=("reservations", "warehouse_sync"),
        postgres_db="inventory",
        tables=("stock_levels", "reservations", "warehouses"),
        redis_cluster="redis-inventory",
        redis_usage="short-lived stock-level cache",
        produces=("inventory.reserved", "inventory.depleted"),
        consumes=("order.created", "order.cancelled"),
        http_dependencies=(
            HttpDependency("product-service", SOFT, "validates SKUs on reservation"),
        ),
        external=(
            ExternalProvider(
                "StockSync WMS",
                "warehouse management system (stock deltas)",
                "stocksync_client",
                "https://api.stocksync.example/v2",
            ),
        ),
    ),
    ServiceProfile(
        id="cart-service",
        display_name="Cart Service",
        package="cart_service",
        team="commerce",
        tier=ServiceTier.TIER_0,
        description="Shopping carts stored in Redis; prices carts via product-service and checks availability via inventory-service.",
        port=8085,
        start_version=(2, 6, 0),
        config=ServiceConfig(
            replicas=6,
            cpu="500m",
            memory="512Mi",
            http_timeout_seconds=1.0,
            cache_ttl_seconds=604800,
            p99_latency_slo_ms=200,
            availability_slo="99.95",
        ),
        endpoints=(
            Endpoint("GET", "/v1/carts/{cart_id}", "get_cart", "Fetch a cart with priced lines"),
            Endpoint("POST", "/v1/carts/{cart_id}/items", "add_item", "Add a SKU to the cart"),
            Endpoint(
                "DELETE",
                "/v1/carts/{cart_id}/items/{sku}",
                "remove_item",
                "Remove a SKU from the cart",
            ),
            Endpoint(
                "POST",
                "/v1/carts/{cart_id}/checkout",
                "checkout_cart",
                "Freeze the cart and hand off to order-service",
            ),
        ),
        domain_modules=("cart_store", "cart_pricing"),
        redis_cluster="redis-carts",
        redis_usage="primary cart store (hash per cart, 7-day TTL)",
        http_dependencies=(
            HttpDependency("product-service", HARD, "current SKU prices"),
            HttpDependency("inventory-service", SOFT, "availability hints"),
        ),
    ),
    ServiceProfile(
        id="order-service",
        display_name="Order Service",
        package="order_service",
        team="commerce",
        tier=ServiceTier.TIER_0,
        description=(
            "Runs checkout: reserves inventory, authorises payment and records the order. "
            "Coordinates the order saga through Kafka events."
        ),
        port=8086,
        start_version=(3, 1, 0),
        config=ServiceConfig(
            replicas=6,
            cpu="1000m",
            memory="1Gi",
            db_pool_size=25,
            db_max_overflow=10,
            db_pool_timeout_seconds=3.0,
            http_timeout_seconds=3.0,
            http_max_retries=1,
            kafka_max_poll_records=200,
            consumer_concurrency=6,
            p99_latency_slo_ms=800,
            availability_slo="99.95",
        ),
        endpoints=(
            Endpoint(
                "POST", "/v1/orders", "create_order", "Place an order from a checked-out cart"
            ),
            Endpoint("GET", "/v1/orders/{order_id}", "get_order", "Fetch an order"),
            Endpoint(
                "POST", "/v1/orders/{order_id}/cancel", "cancel_order", "Cancel an unshipped order"
            ),
        ),
        domain_modules=("checkout", "order_saga"),
        postgres_db="orders",
        tables=("orders", "order_items", "order_events"),
        produces=("order.created", "order.cancelled"),
        consumes=("payment.completed", "payment.failed", "inventory.reserved"),
        http_dependencies=(
            HttpDependency("payment-service", HARD, "authorises and captures payments"),
            HttpDependency("inventory-service", HARD, "reserves stock"),
            HttpDependency("cart-service", HARD, "loads the checked-out cart"),
            HttpDependency("user-service", SOFT, "shipping address lookup"),
        ),
    ),
    ServiceProfile(
        id="payment-service",
        display_name="Payment Service",
        package="payment_service",
        team="payments",
        tier=ServiceTier.TIER_0,
        description=(
            "Authorises, captures and refunds card payments through the PayFlux processor, "
            "with RiskShield fraud scoring, idempotency keys and a double-entry ledger."
        ),
        port=8087,
        start_version=(2, 4, 0),
        config=ServiceConfig(
            replicas=6,
            cpu="1000m",
            memory="1Gi",
            db_pool_size=20,
            db_max_overflow=10,
            db_pool_timeout_seconds=3.0,
            http_timeout_seconds=2.5,
            http_max_retries=2,
            cache_ttl_seconds=86400,
            p99_latency_slo_ms=1200,
            availability_slo="99.95",
        ),
        endpoints=(
            Endpoint("POST", "/v1/payments", "create_payment", "Authorise a card payment"),
            Endpoint(
                "POST",
                "/v1/payments/{payment_id}/capture",
                "capture_payment",
                "Capture an authorised payment",
            ),
            Endpoint("GET", "/v1/payments/{payment_id}", "get_payment", "Fetch payment status"),
            Endpoint("POST", "/v1/refunds", "create_refund", "Refund a captured payment"),
        ),
        domain_modules=("processor", "idempotency", "refunds"),
        postgres_db="payments",
        tables=("payments", "refunds", "idempotency_keys", "ledger_entries"),
        redis_cluster="redis-payments",
        redis_usage="idempotency-key cache (24h TTL)",
        produces=("payment.completed", "payment.failed", "refund.issued"),
        external=(
            ExternalProvider(
                "PayFlux",
                "card payment processor",
                "payflux_client",
                "https://api.payflux.example/v3",
            ),
            ExternalProvider(
                "RiskShield",
                "fraud scoring",
                "riskshield_client",
                "https://score.riskshield.example/v1",
            ),
        ),
        access_level=AccessLevel.SRE,
    ),
    ServiceProfile(
        id="notification-service",
        display_name="Notification Service",
        package="notification_service",
        team="engagement",
        tier=ServiceTier.TIER_2,
        description="Sends transactional email (MailRelay) and SMS (TextBridge) in response to order and payment events.",
        port=8088,
        start_version=(1, 6, 0),
        config=ServiceConfig(
            replicas=3,
            cpu="500m",
            memory="512Mi",
            db_pool_size=10,
            db_max_overflow=5,
            http_timeout_seconds=5.0,
            http_max_retries=3,
            kafka_max_poll_records=500,
            consumer_concurrency=12,
            p99_latency_slo_ms=2000,
            availability_slo="99.5",
        ),
        endpoints=(
            Endpoint(
                "POST", "/v1/notifications", "send_notification", "Send an ad-hoc notification"
            ),
            Endpoint(
                "GET", "/v1/notifications/{notification_id}", "get_notification", "Delivery status"
            ),
        ),
        domain_modules=("dispatcher", "template_renderer"),
        postgres_db="notifications",
        tables=("notifications", "delivery_attempts", "templates"),
        consumes=("order.created", "payment.completed", "payment.failed", "refund.issued"),
        external=(
            ExternalProvider(
                "MailRelay",
                "transactional email",
                "mailrelay_client",
                "https://api.mailrelay.example/v1",
            ),
            ExternalProvider(
                "TextBridge",
                "SMS gateway",
                "textbridge_client",
                "https://api.textbridge.example/v2",
            ),
        ),
    ),
    ServiceProfile(
        id="recommendation-service",
        display_name="Recommendation Service",
        package="recommendation_service",
        team="discovery",
        tier=ServiceTier.TIER_2,
        description=(
            "Personalised and similar-item recommendations from precomputed features; "
            "falls back to popular items when features are unavailable."
        ),
        port=8089,
        start_version=(1, 3, 0),
        config=ServiceConfig(
            replicas=4,
            cpu="1500m",
            memory="2Gi",
            http_timeout_seconds=0.8,
            http_max_retries=0,
            cache_ttl_seconds=3600,
            kafka_max_poll_records=1000,
            consumer_concurrency=4,
            p99_latency_slo_ms=150,
            availability_slo="99.5",
        ),
        endpoints=(
            Endpoint(
                "GET",
                "/v1/recommendations/{user_id}",
                "recommend_for_user",
                "Personalised recommendations",
            ),
            Endpoint(
                "GET",
                "/v1/recommendations/similar/{product_id}",
                "similar_products",
                "Similar items",
            ),
        ),
        domain_modules=("ranker", "feature_store"),
        redis_cluster="redis-reco",
        redis_usage="feature vectors and precomputed recommendation lists",
        consumes=("product.updated", "order.created"),
        http_dependencies=(
            HttpDependency("product-service", SOFT, "hydrates recommended product ids"),
            HttpDependency("user-service", SOFT, "user segment lookup"),
        ),
    ),
    ServiceProfile(
        id="search-service",
        display_name="Search Service",
        package="search_service",
        team="discovery",
        tier=ServiceTier.TIER_1,
        description="Full-text product search and autocomplete backed by the search-products OpenSearch cluster.",
        port=8090,
        start_version=(2, 2, 0),
        config=ServiceConfig(
            replicas=6,
            cpu="1000m",
            memory="1Gi",
            http_timeout_seconds=1.0,
            http_max_retries=1,
            cache_ttl_seconds=120,
            kafka_max_poll_records=500,
            consumer_concurrency=8,
            p99_latency_slo_ms=300,
        ),
        endpoints=(
            Endpoint("GET", "/v1/search", "search_products", "Full-text product search"),
            Endpoint("GET", "/v1/search/suggest", "suggest", "Autocomplete suggestions"),
        ),
        domain_modules=("query_builder", "indexer"),
        redis_cluster="redis-search",
        redis_usage="query-result cache",
        consumes=("product.updated", "price.changed"),
        http_dependencies=(HttpDependency("product-service", SOFT, "full product reindex"),),
        extra_datastores=("opensearch:search-products",),
    ),
)

SERVICES_BY_ID: dict[str, ServiceProfile] = {service.id: service for service in SERVICES}

# Kafka topic -> producing service.
TOPIC_PRODUCERS: dict[str, str] = {
    topic: service.id for service in SERVICES for topic in service.produces
}

TOPIC_PARTITIONS: dict[str, int] = {
    "order.created": 24,
    "order.cancelled": 12,
    "payment.completed": 24,
    "payment.failed": 12,
    "refund.issued": 6,
    "inventory.reserved": 24,
    "inventory.depleted": 6,
    "product.updated": 12,
    "price.changed": 12,
    "user.updated": 6,
    "user.logged_in": 12,
}


def kafka_dependencies(profile: ServiceProfile) -> list[tuple[str, str]]:
    """(producer service, topic) pairs this service consumes from."""
    return [(TOPIC_PRODUCERS[topic], topic) for topic in profile.consumes]


def dependents_of(service_id: str) -> list[ServiceProfile]:
    """Services with a HARD HTTP dependency on ``service_id`` (who breaks when it breaks)."""
    return [
        s
        for s in SERVICES
        if any(d.service == service_id and d.criticality == HARD for d in s.http_dependencies)
    ]


def consumers_of(service_id: str) -> list[ServiceProfile]:
    return [s for s in SERVICES if any(TOPIC_PRODUCERS[t] == service_id for t in s.consumes)]


# --- People -------------------------------------------------------------------


@dataclass(frozen=True)
class RoleSpec:
    """A role. What it may read is defined by the access policy
    (``app/security/policy.json``), not by the dataset."""

    id: str
    name: str
    description: str


ROLES: tuple[RoleSpec, ...] = (
    RoleSpec(
        "developer",
        "Developer",
        "Product engineers: engineering documentation, source code, non-sensitive incidents.",
    ),
    RoleSpec(
        "sre",
        "Site Reliability Engineer",
        "On-call responders: engineering docs, incidents, logs, runbooks, deployments.",
    ),
    RoleSpec(
        "manager",
        "Manager",
        "Engineering and support management: operational reports, incidents, "
        "non-sensitive documents.",
    ),
    RoleSpec(
        "admin",
        "Administrator",
        "Platform and security administrators: all project data the assistant may use.",
    ),
)


@dataclass(frozen=True)
class Person:
    username: str
    full_name: str
    team: str
    role: str
    active: bool = True


PEOPLE: tuple[Person, ...] = (
    Person("arjun.mehta", "Arjun Mehta", "platform", "developer"),
    Person("sofia.rossi", "Sofia Rossi", "platform", "developer"),
    Person("daniel.kim", "Daniel Kim", "platform", "developer"),
    Person("leila.haddad", "Leila Haddad", "platform", "developer"),
    Person("maya.patel", "Maya Patel", "identity", "developer"),
    Person("tom.becker", "Tom Becker", "identity", "developer"),
    Person("grace.okafor", "Grace Okafor", "identity", "developer"),
    Person("ivan.petrov", "Ivan Petrov", "identity", "developer"),
    Person("nina.alvarez", "Nina Alvarez", "catalog", "developer"),
    Person("kenji.watanabe", "Kenji Watanabe", "catalog", "developer"),
    Person("olivia.brown", "Olivia Brown", "catalog", "developer"),
    Person("ravi.iyer", "Ravi Iyer", "discovery", "developer"),
    Person("emma.larsen", "Emma Larsen", "discovery", "developer"),
    Person("lucas.silva", "Lucas Silva", "discovery", "developer"),
    Person("hannah.cho", "Hannah Cho", "discovery", "developer"),
    Person("chen.wei", "Chen Wei", "fulfillment", "developer"),
    Person("fatima.zahra", "Fatima Zahra", "fulfillment", "developer"),
    Person("peter.novak", "Peter Novak", "fulfillment", "developer"),
    Person("aisha.bello", "Aisha Bello", "commerce", "developer"),
    Person("marco.conti", "Marco Conti", "commerce", "developer"),
    Person("julia.schmidt", "Julia Schmidt", "commerce", "developer"),
    Person("samuel.adeyemi", "Samuel Adeyemi", "commerce", "developer"),
    Person("priya.sharma", "Priya Sharma", "payments", "developer"),
    Person("david.cohen", "David Cohen", "payments", "developer"),
    Person("elena.popescu", "Elena Popescu", "payments", "developer"),
    Person("omar.farouk", "Omar Farouk", "payments", "developer"),
    Person("zoe.martin", "Zoe Martin", "engagement", "developer"),
    Person("yusuf.demir", "Yusuf Demir", "engagement", "developer"),
    Person("clara.nguyen", "Clara Nguyen", "engagement", "developer"),
    Person("alex.rivera", "Alex Rivera", "sre", "sre"),
    Person("mei.lin", "Mei Lin", "sre", "sre"),
    Person("jonas.berg", "Jonas Berg", "sre", "sre"),
    Person("kavya.reddy", "Kavya Reddy", "sre", "sre"),
    Person("liam.oconnor", "Liam O'Connor", "sre", "sre"),
    Person("noor.hassan", "Noor Hassan", "security", "admin"),
    Person("erik.johansson", "Erik Johansson", "security", "admin"),
    Person("rosa.garcia", "Rosa Garcia", "support", "manager"),
    Person("ben.taylor", "Ben Taylor", "support", "manager"),
    Person("sarah.miller", "Sarah Miller", "leadership", "manager"),
    # Contract ended: the account stays for audit history but cannot sign in.
    Person("contractor.docs", "Docs Contractor", "external", "developer", active=False),
)


def team_members(team: str) -> list[Person]:
    return [person for person in PEOPLE if person.team == team]


# --- Alerts -------------------------------------------------------------------


def alert_names(profile: ServiceProfile) -> dict[str, str]:
    """Alert rules defined for a service (keys are stable identifiers)."""
    c = profile.camel
    alerts = {
        "http_5xx": f"{c}High5xxRate",
        "latency": f"{c}LatencyP99High",
        "restarts": f"{c}PodRestartsHigh",
        "memory": f"{c}MemoryNearLimit",
        "cpu": f"{c}CPUThrottlingHigh",
        "upstream": f"{c}UpstreamErrorRate",
    }
    if profile.postgres_db:
        alerts["db_pool"] = f"{c}DBPoolSaturated"
        alerts["db_deadlock"] = f"{c}DBDeadlocksDetected"
        alerts["db_errors"] = f"{c}DBQueryErrors"
    if profile.redis_cluster:
        alerts["redis_memory"] = f"{c}RedisMemoryHigh"
        alerts["cache_hit"] = f"{c}CacheHitRatioLow"
    if profile.consumes:
        alerts["consumer_lag"] = f"{c}ConsumerLagHigh"
        alerts["consumer_errors"] = f"{c}ConsumerErrorRate"
    if profile.external:
        alerts["provider"] = f"{c}ProviderErrorRate"
    if profile.id in {"api-gateway", "auth-service"}:
        alerts["auth_failures"] = f"{c}AuthFailuresHigh"
    if profile.id == "api-gateway":
        alerts["rate_limited"] = "ApiGateway429RateHigh"
    return alerts
