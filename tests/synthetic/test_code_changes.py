"""Faulty and maintenance changes are real diffs against the rendered repository."""

from __future__ import annotations

import random

import pytest

from app.synthetic.catalog import SERVICES
from app.synthetic.changes import FAULT_BUILDERS, build_fault, choose_maintenance
from app.synthetic.code_repo import build_repository
from app.synthetic.domain_code import (
    AUTH_FAULTS,
    GATEWAY_RATE_LIMIT_FAULT,
    REGRESSIONS,
    domain_module_source,
)

REPO = build_repository()


@pytest.mark.parametrize(
    "fault",
    [*REGRESSIONS.values(), *AUTH_FAULTS.values(), GATEWAY_RATE_LIMIT_FAULT],
    ids=lambda f: f"{f.service_id}:{f.module}",
)
def test_hand_written_fault_snippets_match_exactly_once(fault) -> None:
    source = domain_module_source(fault.service_id, fault.module)
    for old, _ in fault.replacements:
        assert source.count(old) == 1, old


@pytest.mark.parametrize("service", SERVICES, ids=lambda s: s.id)
def test_faults_diff_from_head_and_fixes_restore_it(service) -> None:
    applicable = 0
    for kind in FAULT_BUILDERS:
        plan = build_fault(kind, service, REPO, random.Random(3))
        if plan is None:
            continue
        applicable += 1
        for edit in plan.edits:
            assert edit.before == REPO[edit.path].content
            assert edit.after != edit.before and edit.patch.startswith(f"--- a/{edit.path}")
        for fix in plan.fix_edits:
            assert fix.after == REPO[fix.path].content
    assert applicable >= 5


@pytest.mark.parametrize("service", SERVICES, ids=lambda s: s.id)
def test_maintenance_changes_end_at_head(service) -> None:
    rng = random.Random(11)
    for _ in range(25):
        change = choose_maintenance(service, REPO, rng)
        for edit in change.edits:
            assert edit.after == REPO[edit.path].content and edit.patch
