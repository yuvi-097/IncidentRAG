"""Detection and redaction of credentials in text.

Applied to retrieved content before it reaches a model, to final answers, and to
every log line. Two kinds of rules:
- well-known token formats (private keys, provider API keys, JWTs, bearer tokens,
  credentials embedded in URLs);
- assignments whose name says "secret" with a literal value (``DB_PASSWORD=...``,
  ``"api_key": "..."``), unless the value is a reference to a secret rather than the
  secret itself (``${VAULT_X}``, ``vault://...``).
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

_TOKENS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----.*?"
            r"(?:-----END (?:[A-Z]+ )*PRIVATE KEY-----|\Z)",
            re.S,
        ),
    ),
    ("opsrag_token", re.compile(r"\bopsrag_[0-9a-f]{12}_[A-Za-z0-9_-]{43}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|live-|test-|svcacct-)?[A-Za-z0-9_-]{20,}")),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk|whsec)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*")),
)
# scheme://user:PASSWORD@host
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s@/]+)(@)")
# A name that says "secret", ending in the secret word (so ``password_hash`` or
# ``token_type`` do not count).
_SECRET_NAME = (
    r"(?<![A-Za-z0-9_])[A-Za-z0-9_.-]*?(?:password|passwd|pwd|secret|api[_-]?key|apikey|"
    r"access[_-]?key|auth[_-]?token|access[_-]?token|refresh[_-]?token|client[_-]?secret|"
    r"private[_-]?key|signing[_-]?key)(?![A-Za-z0-9_])"
)
# name = "literal", "name": "literal", name: 'literal' (JSON, Python, YAML, logs)
_QUOTED_ASSIGNMENT = re.compile(
    rf"(?i)({_SECRET_NAME}[\"']?\s*[:=]\s*)([\"'])([^\"'\n]{{6,}}?)(\2)"
)
# NAME=value with nothing around "=" (.env files, command lines, query strings).
# Unquoted code such as ``"access_token": access`` is a variable, not a secret, and
# neither is a call or an attribute path (``password=self.password.get()``).
_ENV_ASSIGNMENT = re.compile(rf"(?i)({_SECRET_NAME}=)([^\s\"'`,;&{{}}<>()\[\]\\]{{6,}}+)(?!\()")
_ATTRIBUTE_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
# A constant passed as the value: a name, not a secret.
_CONSTANT_NAME = re.compile(r"[A-Z]+(?:_[A-Z]+)+")
# Values that point at a secret instead of containing one.
_REFERENCE = re.compile(
    r"(?i)^(?:\$|%|vault:|secret/|secrets/|env:|ref:|arn:|projects/|\[redacted|\*{3,}|x{6,}|"
    r"<|none$|null$|true$|false$|changeme$|example)"
)
_PLACEHOLDER = re.compile(r"^[A-Z_]+$")  # user:PASSWORD@host in documentation
# Log/extra field names whose values are always secret.
SECRET_FIELD = re.compile(
    r"(?i)^(?:.*[_-])?(?:password|passwd|secret|api[_-]?key|apikey|token|access[_-]?token|"
    r"auth[_-]?token|refresh[_-]?token|authorization|cookie|credentials?|private[_-]?key|"
    r"client[_-]?secret)$"
)


def redact_secrets(text: str) -> tuple[str, int]:
    """``text`` with every credential replaced by ``[REDACTED]``, and how many."""
    if not text:
        return text, 0
    count = 0
    for _, pattern in _TOKENS:
        text, n = pattern.subn(REDACTED, text)
        count += n

    def url(match: re.Match[str]) -> str:
        nonlocal count
        password = match.group(2)
        if _REFERENCE.match(password) or _PLACEHOLDER.match(password) or password == REDACTED:
            return match.group(0)
        count += 1
        return f"{match.group(1)}{REDACTED}{match.group(3)}"

    text = _URL_CREDENTIALS.sub(url, text)

    def quoted(match: re.Match[str]) -> str:
        nonlocal count
        if _REFERENCE.match(match.group(3)) or match.group(3) == REDACTED:
            return match.group(0)
        count += 1
        return f"{match.group(1)}{match.group(2)}{REDACTED}{match.group(4)}"

    def env(match: re.Match[str]) -> str:
        nonlocal count
        value = match.group(2)
        if (
            _REFERENCE.match(value)
            or _ATTRIBUTE_PATH.fullmatch(value)
            or _CONSTANT_NAME.fullmatch(value)
            or value == REDACTED
        ):
            return match.group(0)
        count += 1
        return f"{match.group(1)}{REDACTED}"

    text = _QUOTED_ASSIGNMENT.sub(quoted, text)
    text = _ENV_ASSIGNMENT.sub(env, text)
    return text, count


def contains_secret(text: str) -> bool:
    return redact_secrets(text)[1] > 0


__all__ = ["REDACTED", "SECRET_FIELD", "contains_secret", "redact_secrets"]
