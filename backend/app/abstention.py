from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

TypedAbstentionReason = Literal[
    "uncertified_due_to_search",
    "uncertified_due_to_evidence",
    "uncertified_due_to_preference",
    "uncertified_due_to_out_of_support_world_model",
    "uncertified_due_to_budget",
    "uncertified_due_to_model_assumption",
]


class AbstentionRecord(BaseModel):
    """Typed abstention payload used by the certification wrapper."""

    reason_code: TypedAbstentionReason
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    support_flag: bool | None = None
    evidence_family: str | None = None
    budget_channel: str | None = None
    model_assumption: str | None = None
    terminal_type: Literal["typed_abstention"] = "typed_abstention"


_ABSTENTION_MESSAGES: dict[TypedAbstentionReason, str] = {
    "uncertified_due_to_search": "Terminal decision stopped before a certified singleton was justified.",
    "uncertified_due_to_evidence": "Evidence support remained insufficient for certification.",
    "uncertified_due_to_preference": "Preference ambiguity prevented singleton certification.",
    "uncertified_due_to_out_of_support_world_model": "Selected support state was out of support for certification.",
    "uncertified_due_to_budget": "Budget was exhausted before certification completed.",
    "uncertified_due_to_model_assumption": "Certification assumptions were not strong enough for a terminal decision.",
}


def _as_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    return any(token in text for token in tokens)


def classify_typed_abstention_reason(
    *,
    stop_reason: str | None = None,
    support_flag: bool | None = None,
    support_reason: str | None = None,
    credible_search_uncertainty: bool | None = None,
    credible_evidence_uncertainty: bool | None = None,
    search_completeness_score: float | None = None,
    search_completeness_gap: float | None = None,
    evidence_family: str | None = None,
    budget_channel: str | None = None,
    model_assumption: str | None = None,
    active_families: Sequence[str] | None = None,
    top_fragility_families: Sequence[str] | None = None,
) -> TypedAbstentionReason:
    """Map existing stop/support signals to a typed abstention reason.

    The mapping is intentionally conservative: budget and out-of-support signals
    take precedence, then search/evidence/preference signals, and model
    assumption is the fallback when no more specific explanation is present.
    """

    stop_text = _as_text(stop_reason)
    support_text = _as_text(support_reason)
    model_text = _as_text(model_assumption)
    budget_text = _as_text(budget_channel)
    evidence_text = _as_text(evidence_family)
    active_family_count = len([family for family in (active_families or []) if _as_text(family)])
    fragility_family_count = len([family for family in (top_fragility_families or []) if _as_text(family)])

    if budget_text or _contains_any(stop_text, ("budget", "exhausted", "quota")):
        return "uncertified_due_to_budget"

    if support_flag is False or _contains_any(
        support_text,
        ("out_of_support", "unsupported", "outside_support", "out of support"),
    ):
        return "uncertified_due_to_out_of_support_world_model"

    if credible_search_uncertainty is True or _contains_any(
        stop_text,
        ("search_incomplete", "no_action_worth_it", "search_uncertain", "insufficient_search"),
    ) or (search_completeness_score is not None and search_completeness_score < 0.5) or (
        search_completeness_gap is not None and search_completeness_gap > 0.0
    ):
        return "uncertified_due_to_search"

    if credible_evidence_uncertainty is True or evidence_text or fragility_family_count > 0 or _contains_any(
        stop_text,
        ("evidence", "fragility", "refresh", "audit"),
    ):
        return "uncertified_due_to_evidence"

    if _contains_any(support_text, ("preference", "veto", "ratio", "threshold")) or _contains_any(
        stop_text,
        ("preference", "query", "elicitation"),
    ):
        return "uncertified_due_to_preference"

    if model_text or (active_family_count == 0 and support_flag is not None):
        return "uncertified_due_to_model_assumption"

    return "uncertified_due_to_model_assumption"


def build_abstention_record(
    *,
    stop_reason: str | None = None,
    support_flag: bool | None = None,
    support_reason: str | None = None,
    credible_search_uncertainty: bool | None = None,
    credible_evidence_uncertainty: bool | None = None,
    search_completeness_score: float | None = None,
    search_completeness_gap: float | None = None,
    evidence_family: str | None = None,
    budget_channel: str | None = None,
    model_assumption: str | None = None,
    active_families: Sequence[str] | None = None,
    top_fragility_families: Sequence[str] | None = None,
    detail: dict[str, Any] | None = None,
) -> AbstentionRecord:
    reason_code = classify_typed_abstention_reason(
        stop_reason=stop_reason,
        support_flag=support_flag,
        support_reason=support_reason,
        credible_search_uncertainty=credible_search_uncertainty,
        credible_evidence_uncertainty=credible_evidence_uncertainty,
        search_completeness_score=search_completeness_score,
        search_completeness_gap=search_completeness_gap,
        evidence_family=evidence_family,
        budget_channel=budget_channel,
        model_assumption=model_assumption,
        active_families=active_families,
        top_fragility_families=top_fragility_families,
    )
    payload_detail = dict(detail or {})
    payload_detail.setdefault("stop_reason", stop_reason)
    payload_detail.setdefault("support_reason", support_reason)
    payload_detail.setdefault("search_completeness_score", search_completeness_score)
    payload_detail.setdefault("search_completeness_gap", search_completeness_gap)
    payload_detail.setdefault("active_families", list(active_families or []))
    payload_detail.setdefault("top_fragility_families", list(top_fragility_families or []))
    return AbstentionRecord(
        reason_code=reason_code,
        message=_ABSTENTION_MESSAGES[reason_code],
        detail=payload_detail,
        support_flag=support_flag,
        evidence_family=evidence_family,
        budget_channel=budget_channel,
        model_assumption=model_assumption,
    )
