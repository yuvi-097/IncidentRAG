"""Static safety checks for agent-written SQL. Runs before anything reaches the database.

The policy is deliberately narrow; anything the checker is unsure about is rejected:

- exactly one statement, starting with ``SELECT`` or ``WITH``;
- no data-changing or session keywords anywhere (INSERT, UPDATE, DELETE, DROP, ALTER,
  TRUNCATE, CREATE, GRANT, COPY, SET, INTO, ...), checked on tokens, so the same
  words inside string literals are fine;
- constructs where database parsers and this lexer could disagree are rejected
  outright: comments, dollar-quoting, prefixed strings (``E'...'``), backslashes in
  strings, bind parameters, backtick/bracket identifiers;
- functions only from an allowlist (blocks ``pg_sleep``, ``pg_read_file``,
  ``query_to_xml`` and anything that runs SQL from a string);
- relations only from an allowlist of tables plus CTEs the query defines itself;
  schema-qualified names (``public.users``, ``main.incidents``, ``pg_catalog.x``) are
  rejected, so the access-filtered views the executor puts in front of every table
  cannot be bypassed.

The executor (``sql_tool``) adds the runtime layers: a read-only transaction, a
statement timeout, a row limit, and per-caller filtered views of each table.
"""

from __future__ import annotations

import re
import string
from collections.abc import Collection
from dataclasses import dataclass

from app.tools.base import UnsafeSqlError


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


MAX_SQL_CHARS = 5000
MAX_TOKENS = 2000
MAX_DEPTH = 32

FORBIDDEN_KEYWORDS = _words(
    """
    insert update delete drop alter truncate create replace merge upsert grant revoke
    copy call do execute exec prepare deallocate vacuum analyze analyse reindex cluster
    refresh lock listen unlisten notify comment set reset begin commit rollback
    savepoint release start transaction attach detach pragma into recursive share nowait
    returning only table load import discard checkpoint security owner cursor fetch
    declare
    """
)
# ``replace(...)`` is also a harmless string function; it is allowed only as a call.
_CALLABLE_KEYWORDS = frozenset({"replace"})

ALLOWED_FUNCTIONS = _words(
    """
    count sum avg min max coalesce nullif greatest least abs round ceil ceiling floor
    trunc sign mod power sqrt lower upper length char_length substr substring trim ltrim
    rtrim replace concat strpos instr split_part left right lpad rpad reverse initcap
    date_trunc date_part extract to_char to_date to_timestamp age now date datetime
    strftime julianday unixepoch make_interval date_bin cast string_agg group_concat
    array_agg json_agg jsonb_agg bool_and bool_or every percentile_cont percentile_disc
    mode stddev stddev_pop stddev_samp variance row_number rank dense_rank percent_rank
    cume_dist ntile lag lead first_value last_value nth_value json_extract
    json_array_length jsonb_array_length jsonb_array_elements_text
    jsonb_extract_path_text json_extract_path_text jsonb_typeof json_type iif ifnull
    generate_series unnest
    """
)
TABLE_FUNCTIONS = frozenset({"generate_series", "unnest", "jsonb_array_elements_text"})

# SQL keywords that may be followed by "(" without being a function call.
_KEYWORDS_BEFORE_PAREN = _words(
    """
    select from where and or not in exists any all some as on using join having by over
    filter within values when then else case union intersect except distinct lateral
    like ilike between is limit offset materialized row array interval
    """
)
_BLOCKED_IDENTIFIER_PREFIXES = ("pg_", "sqlite_", "information_schema", "lo_", "dblink")
_BLOCKED_IDENTIFIERS = frozenset(
    {"public", "main", "temp", "temporary", "current_setting", "set_config", "query_to_xml"}
)
# Words that end a FROM list (JOIN ... ON ... keeps it open: "FROM a JOIN b ON ..., c").
_CLAUSE_WORDS = _words(
    """
    where group order having limit offset window union intersect except select
    """
)

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_OPERATORS = ("::", "->>", "->", "<=", ">=", "<>", "!=", "||", "~~")
_SINGLE_CHARS = set("(),.*+-/%=<>;|~@&#^")
_WORD_START = set(string.ascii_letters + "_")
_DIGITS = set(string.digits)


@dataclass(frozen=True)
class Token:
    kind: str  # word | ident (double-quoted) | string | number | op
    value: str  # words and idents lower-cased; the original text for others
    position: int


@dataclass(frozen=True)
class CheckedSql:
    """A query that passed every static check."""

    sql: str  # the original text without a trailing semicolon
    tables: frozenset[str]  # allowed tables it reads
    ctes: tuple[str, ...]  # CTE names it defines (in order)
    starts_with_with: bool


def _reject(reason: str) -> UnsafeSqlError:
    return UnsafeSqlError(f"unsafe SQL: {reason}", [{"reason": reason}])


def tokenize(sql: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
        elif sql.startswith("--", i) or sql.startswith("/*", i):
            raise _reject("comments are not allowed")
        elif ch == "'":
            end = i + 1
            while True:
                end = sql.find("'", end)
                if end == -1:
                    raise _reject("unterminated string literal")
                if sql.startswith("''", end):
                    end += 2
                    continue
                break
            literal = sql[i : end + 1]
            if "\\" in literal:
                raise _reject("backslashes are not allowed in string literals")
            tokens.append(Token("string", literal, i))
            i = end + 1
        elif ch == '"':
            end = sql.find('"', i + 1)
            name = sql[i + 1 : end] if end != -1 else ""
            if end == -1 or not _WORD.fullmatch(name):
                raise _reject("quoted identifiers may only contain letters, digits and _")
            tokens.append(Token("ident", name.lower(), i))
            i = end + 1
        elif ch in _WORD_START:
            match = _WORD.match(sql, i)
            assert match is not None
            word = match.group()
            if match.end() < n and sql[match.end()] == "'":
                raise _reject(f"prefixed string literals ({word}'...') are not allowed")
            tokens.append(Token("word", word.lower(), i))
            i = match.end()
        elif ch in _DIGITS or (ch == "." and i + 1 < n and sql[i + 1] in _DIGITS):
            match = _NUMBER.match(sql, i)
            assert match is not None
            tokens.append(Token("number", match.group(), i))
            i = match.end()
        else:
            operator = next((op for op in _OPERATORS if sql.startswith(op, i)), None)
            if operator:
                tokens.append(Token("op", operator, i))
                i += len(operator)
            elif ch in _SINGLE_CHARS:
                tokens.append(Token("op", ch, i))
                i += 1
            else:
                raise _reject(f"character {ch!r} is not allowed")
        if len(tokens) > MAX_TOKENS:
            raise _reject("query is too long")
    return tokens


def _is_name(token: Token | None) -> bool:
    return token is not None and token.kind in {"word", "ident"}


def _defined_ctes(tokens: list[Token]) -> list[str]:
    """Names in the leading ``WITH a AS (...), b AS (...)`` list."""
    if not tokens or tokens[0].value != "with":
        return []
    names, i, depth = [], 1, 0
    expect_name = True
    while i < len(tokens):
        token = tokens[i]
        if token.value == "(":
            depth += 1
        elif token.value == ")":
            depth -= 1
        elif depth == 0:
            if expect_name:
                if not _is_name(token):
                    raise _reject("malformed WITH clause")
                nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                if nxt is None or nxt.value != "as":
                    raise _reject("CTE column lists are not supported; alias columns inside")
                names.append(token.value)
                expect_name = False
            elif token.value == ",":
                expect_name = True
            elif token.value == "select":
                break
        i += 1
    return names


def check_sql(
    sql: str, allowed_tables: Collection[str], reserved_names: Collection[str] = ()
) -> CheckedSql:
    """Raise ``UnsafeSqlError`` unless ``sql`` is a single safe read-only query.

    ``allowed_tables`` may be read; ``reserved_names`` (every table in the schema) may
    not be used as CTE names, because the executor defines CTEs with those names."""
    text = sql.strip()
    if not text:
        raise _reject("empty query")
    if len(text) > MAX_SQL_CHARS:
        raise _reject(f"query longer than {MAX_SQL_CHARS} characters")
    tokens = tokenize(text)
    if tokens and tokens[-1].value == ";":
        tokens.pop()
        text = text[: text.rstrip().rfind(";")].rstrip()
    if not tokens:
        raise _reject("empty query")
    if any(t.value == ";" for t in tokens):
        raise _reject("only one statement is allowed")
    for index, token in enumerate(tokens):
        if token.kind == "word" and token.value in FORBIDDEN_KEYWORDS:
            nxt = tokens[index + 1] if index + 1 < len(tokens) else None
            if not (token.value in _CALLABLE_KEYWORDS and nxt is not None and nxt.value == "("):
                raise _reject(f"keyword {token.value.upper()} is not allowed")
    if tokens[0].kind != "word" or tokens[0].value not in {"select", "with"}:
        raise _reject("only SELECT queries (optionally starting with WITH) are allowed")

    allowed = {t.lower() for t in allowed_tables}
    reserved = allowed | {t.lower() for t in reserved_names}
    ctes = _defined_ctes(tokens)
    for name in ctes:
        if name in reserved or name in FORBIDDEN_KEYWORDS:
            raise _reject(f"CTE name {name!r} is reserved")
    relations = allowed | set(ctes)
    referenced: set[str] = set()

    depth = 0
    from_depths: list[int] = []  # depths at which a FROM list is open
    calls: list[bool] = []  # for each open "(": is it a function call's argument list?
    expect_relation = False
    for index, token in enumerate(tokens):
        nxt = tokens[index + 1] if index + 1 < len(tokens) else None
        prev = tokens[index - 1] if index else None
        value = token.value
        if _is_name(token):
            if value in _BLOCKED_IDENTIFIERS or value.startswith(_BLOCKED_IDENTIFIER_PREFIXES):
                raise _reject(f"identifier {value!r} is not allowed")
            if nxt is not None and nxt.value == "(" and value not in _KEYWORDS_BEFORE_PAREN:
                if value not in ALLOWED_FUNCTIONS:
                    raise _reject(f"function {value}() is not allowed")
                if expect_relation and value not in TABLE_FUNCTIONS:
                    raise _reject(f"function {value}() cannot be used as a table")
                expect_relation = False
                continue
        if value == "(":
            depth += 1
            if depth > MAX_DEPTH:
                raise _reject("query is nested too deeply")
            is_call = _is_name(prev) and prev is not None and prev.value in ALLOWED_FUNCTIONS
            calls.append(is_call and prev is not None and prev.value not in _KEYWORDS_BEFORE_PAREN)
            expect_relation = False  # a subquery or table function in FROM position
            continue
        if value == ")":
            depth -= 1
            if depth < 0:
                raise _reject("unbalanced parentheses")
            calls.pop()
            while from_depths and from_depths[-1] > depth:
                from_depths.pop()
            continue
        if expect_relation:
            if not _is_name(token):
                raise _reject("expected a table name after FROM/JOIN")
            if nxt is not None and nxt.value == ".":
                raise _reject("schema-qualified table names are not allowed")
            if value not in relations:
                raise _reject(f"table {value!r} is not available")
            if value in allowed:
                referenced.add(value)
            expect_relation = False
            continue
        if token.kind == "word" and value in {"from", "join"}:
            if value == "from" and prev is not None and prev.value in {"distinct", "is"}:
                continue  # IS [NOT] DISTINCT FROM
            if value == "from" and calls and calls[-1]:
                continue  # EXTRACT(year FROM x), SUBSTRING(x FROM 2), TRIM(... FROM x)
            expect_relation = True
            if value == "from":
                from_depths.append(depth)
            continue
        if value == "," and from_depths and from_depths[-1] == depth:
            expect_relation = True
            continue
        if token.kind == "word" and value in _CLAUSE_WORDS and from_depths[-1:] == [depth]:
            from_depths.pop()
    if depth != 0:
        raise _reject("unbalanced parentheses")
    if expect_relation:
        raise _reject("expected a table name after FROM/JOIN")
    return CheckedSql(text, frozenset(referenced), tuple(ctes), tokens[0].value == "with")
