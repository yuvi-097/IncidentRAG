"""SQL safety: the static checker rejects anything that is not one read-only SELECT
over allowed tables, before execution. Pure tests, no database."""

from __future__ import annotations

import pytest

from app.tools import UnsafeSqlError
from app.tools.sql_guard import FORBIDDEN_KEYWORDS, check_sql, tokenize

ALLOWED = {"incidents", "deployments", "services", "logs", "documents"}
RESERVED = ALLOWED | {"users", "roles", "chunk_embeddings", "document_chunks"}
BS = chr(92)  # a backslash, kept out of literals


def check(sql: str) -> set[str]:
    return set(check_sql(sql, ALLOWED, RESERVED).tables)


def rejected(sql: str) -> str:
    with pytest.raises(UnsafeSqlError) as info:
        check_sql(sql, ALLOWED, RESERVED)
    return info.value.message


@pytest.mark.parametrize(
    ("sql", "tables"),
    [
        ("SELECT count(*) FROM incidents", {"incidents"}),
        ("select * from incidents;", {"incidents"}),
        (
            "SELECT service_id, count(*) AS n FROM incidents WHERE started_at >= '2026-08-01' "
            "GROUP BY service_id ORDER BY n DESC LIMIT 5",
            {"incidents"},
        ),
        (
            "SELECT i.id, d.version FROM incidents i JOIN deployments d ON d.id = i.deployment_id, "
            "services s WHERE s.id = i.service_id",
            {"incidents", "deployments", "services"},
        ),
        (
            "WITH sev1 AS (SELECT * FROM incidents WHERE severity = 'SEV1') "
            "SELECT count(*) FROM sev1",
            {"incidents"},
        ),
        (
            "SELECT extract(year FROM started_at), substring(title FROM 1 FOR 5) FROM incidents",
            {"incidents"},
        ),
        (
            "SELECT id FROM incidents WHERE id IN (SELECT id FROM incidents) "
            "UNION SELECT id FROM deployments",
            {"incidents", "deployments"},
        ),
        ("SELECT coalesce((SELECT max(started_at) FROM incidents), now())", {"incidents"}),
        ("SELECT a IS NOT DISTINCT FROM b FROM incidents", {"incidents"}),
        ('SELECT "timestamp", level FROM logs', {"logs"}),
        ("SELECT metrics->>'peak_error_rate_pct' FROM incidents", {"incidents"}),
        ("SELECT replace(title, 'a', 'b') FROM incidents", {"incidents"}),
        ("SELECT 1", set()),
    ],
)
def test_read_only_queries_are_accepted(sql: str, tables: set[str]) -> None:
    assert check(sql) == tables


@pytest.mark.parametrize(
    "keyword", ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]
)
@pytest.mark.parametrize(
    "template",
    [
        "{kw} incidents",
        "{kw} TABLE incidents",
        "SELECT 1; {kw} TABLE incidents",
        "WITH x AS ({kw} FROM incidents RETURNING *) SELECT * FROM x",
        "SELECT * FROM incidents WHERE id IN ({kw} incidents)",
    ],
)
def test_data_changing_statements_are_blocked(keyword: str, template: str) -> None:
    for spelling in (keyword, keyword.lower(), keyword.title()):
        message = rejected(template.format(kw=spelling))
        assert "not allowed" in message or "one statement" in message


def test_every_forbidden_keyword_is_blocked_as_a_word() -> None:
    for keyword in sorted(FORBIDDEN_KEYWORDS - {"replace"}):
        assert keyword.upper() in rejected(f"SELECT {keyword} FROM incidents").upper()


def test_keywords_inside_string_literals_are_data() -> None:
    assert check("SELECT * FROM incidents WHERE title LIKE '%DROP TABLE users; DELETE%'") == {
        "incidents"
    }
    assert check("SELECT 'it''s; fine' FROM incidents") == {"incidents"}


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("", "empty"),
        ("   ;  ", "empty"),
        ("SELECT 1; SELECT 2", "one statement"),
        ("EXPLAIN SELECT 1", "only SELECT"),
        ("SHOW tables", "only SELECT"),
        ("SELECT * FROM users", "not available"),
        ("SELECT * FROM chunk_embeddings", "not available"),
        (
            "SELECT * FROM incidents i JOIN deployments d ON d.id = i.deployment_id, users u",
            "not available",
        ),
        ("SELECT * FROM public.incidents", "'public'"),
        ("SELECT * FROM main.incidents", "'main'"),
        ("SELECT * FROM pg_catalog.pg_user", "pg_catalog"),
        ("SELECT * FROM information_schema.tables", "information_schema"),
        ("SELECT * FROM sqlite_master", "sqlite_master"),
        ("SELECT pg_sleep(10)", "pg_sleep"),
        ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
        ("SELECT query_to_xml('select * from users', true, true, '')", "query_to_xml"),
        ("SELECT set_config('a', 'b', false)", "set_config"),
        ("SELECT version()", "version()"),
        ("SELECT dblink('x', 'y')", "dblink"),
        ("SELECT * INTO copy_of_incidents FROM incidents", "INTO"),
        ("SELECT * FROM incidents FOR UPDATE", "UPDATE"),
        ("SELECT * FROM incidents FOR SHARE", "SHARE"),
        ("WITH RECURSIVE r AS (SELECT 1) SELECT * FROM r", "RECURSIVE"),
        ("WITH users AS (SELECT 1) SELECT * FROM users", "reserved"),
        ("WITH incidents AS (SELECT 1) SELECT * FROM incidents", "reserved"),
        ("WITH x(a) AS (SELECT 1) SELECT * FROM x", "column lists"),
        ("SELECT * FROM incidents -- trailing comment", "comments"),
        ("SELECT * /* hidden */ FROM incidents", "comments"),
        ("SELECT $$x$$", "'$'"),
        ("SELECT * FROM incidents WHERE id = :id", "':'"),
        ("SELECT * FROM incidents WHERE id = ?", "'?'"),
        ("SELECT * FROM [users]", "'['"),
        ("SELECT * FROM `users`", "'`'"),
        ("SELECT E'x' FROM incidents", "prefixed"),
        ("SELECT * FROM incidents WHERE title = 'a" + BS + "' OR 1=1'", "backslashes"),
        ("SELECT 'unterminated FROM incidents", "unterminated"),
        ("SELECT (1 FROM incidents", "parentheses"),
        ("SELECT * FROM", "table name"),
        ("SELECT * FROM generate_series(1, 3) g, users", "not available"),
        ("SELECT * FROM count(1)", "cannot be used as a table"),
        ('SELECT * FROM "users"', "not available"),
        ('SELECT * FROM "in cidents"', "quoted identifiers"),
        (chr(0xFF24) + "ELETE FROM incidents", "not allowed"),  # full-width D
        ("SELECT 1 FROM incidents WHERE x = " + chr(0x0663), "not allowed"),  # Arabic-Indic 3
    ],
)
def test_unsafe_sql_is_rejected_with_a_reason(sql: str, reason: str) -> None:
    assert reason in rejected(sql)


def test_size_limits() -> None:
    assert "longer than" in rejected("SELECT 1 " + " " * 5000 + "FROM incidents x" * 10)
    assert "nested too deeply" in rejected("SELECT " + "(" * 40 + "1" + ")" * 40)


def test_tokenizer_keeps_strings_and_quoted_identifiers_intact() -> None:
    tokens = tokenize("SELECT \"Level\", 'a''b' FROM logs WHERE x <> 1.5e3")
    assert [t.kind for t in tokens] == [
        "word",
        "ident",
        "op",
        "string",
        "word",
        "word",
        "word",
        "word",
        "op",
        "number",
    ]
    assert tokens[1].value == "level" and tokens[3].value == "'a''b'"
