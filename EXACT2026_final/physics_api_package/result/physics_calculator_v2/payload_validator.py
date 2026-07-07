from __future__ import annotations

from typing import Any, Dict

from .formula_bank import FORMULA_BY_ID, candidate_ids
from .quantity_parser import parse_given, target_role, target_unit


FORBIDDEN_OUTPUT_KEYS = {"answer", "unit_answer", "python_code", "golden_code", "final_result", "cot"}


def validate_numeric_payload(question: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parser JSON before the calculator trusts it.

    This validator is intentionally conservative. It does not decide whether a
    formula will numerically solve; it only catches schema/registry failures
    that should not reach the calculator as if they were valid extractions.
    """

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not isinstance(payload, dict):
        return {"ok": False, "errors": [{"type": "payload_not_object"}], "warnings": []}

    forbidden = sorted(key for key in FORBIDDEN_OUTPUT_KEYS if key in payload)
    if forbidden:
        errors.append({"type": "forbidden_output_keys", "keys": forbidden})

    if not target_role(payload):
        errors.append({"type": "missing_target_role"})
    if not target_unit(payload):
        warnings.append({"type": "missing_target_unit"})

    raw_givens = payload.get("givens")
    if not isinstance(raw_givens, list) or not raw_givens:
        errors.append({"type": "missing_givens"})
        parsed_givens = []
    else:
        parsed_givens = []
        for index, given in enumerate(raw_givens):
            parsed = parse_given(given, question=question)
            if parsed is None:
                warnings.append({"type": "unreadable_given", "index": index, "given": given})
            else:
                parsed_givens.append(parsed)

    parsed_roles = [given["role"] for given in parsed_givens]
    parsed_dimensions = [given["dimension"] for given in parsed_givens]

    unknown_formula_ids = [formula_id for formula_id in candidate_ids(payload) if formula_id not in FORMULA_BY_ID]
    if unknown_formula_ids:
        errors.append({"type": "unknown_formula_id", "formula_ids": unknown_formula_ids})

    for formula_id in candidate_ids(payload):
        spec = FORMULA_BY_ID.get(formula_id)
        if spec is None:
            continue
        missing_roles = []
        for role in set(spec.required_roles):
            required_count = spec.required_roles.count(role)
            available_count = parsed_roles.count(role) + parsed_dimensions.count(role)
            if available_count < required_count:
                missing_roles.append({"role": role, "required": required_count, "available": available_count})
        if missing_roles:
            warnings.append({"type": "formula_missing_roles", "formula_id": formula_id, "missing": missing_roles})

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "parsed_given_count": len(parsed_givens),
        "parsed_roles": parsed_roles,
        "candidate_formula_ids": candidate_ids(payload),
    }
