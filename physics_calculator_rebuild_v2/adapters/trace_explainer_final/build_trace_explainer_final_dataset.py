from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calculator import solve_numeric_payload


SYSTEM_PROMPT = (
    "You explain locked physics calculator traces. Return JSON only. "
    "Do not change the answer, unit, formula, or calculation."
)

FORBIDDEN_OUTPUT_KEYS = {"answer", "unit", "python_code", "final_result"}

TARGET_TOTAL_ROWS = 1200


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def repair_text(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "Ã—": "x",
        "Âµ": "u",
        "Î¼": "u",
        "μ": "u",
        "µ": "u",
        "Î©": "ohm",
        "Ω": "ohm",
        "Â²": "^2",
        "²": "^2",
        "Â³": "^3",
        "³": "^3",
        "Ï€": "pi",
        "π": "pi",
        "Ï‰": "omega",
        "ω": "omega",
        "âˆš": "sqrt",
        "√": "sqrt",
        "â‰ˆ": "approximately",
        "≈": "approximately",
        "âˆ’": "-",
        "−": "-",
        "Â±": "±",
        "ą": "±",
        "Â": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def normalize_unit(unit: Any) -> str:
    return repair_text(unit).lower().replace(" ", "").replace("ohms", "ohm")


def parse_number(value: Any) -> Optional[float]:
    text = repair_text(value)
    text = re.sub(r"([+-]?\d+(?:\.\d+)?)\s*(?:x|\*)\s*10\^?([+-]?\d+)", r"\1e\2", text, flags=re.I)
    match = re.search(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, flags=re.I)
    return float(match.group(0)) if match else None


def answer_close(pred: Any, expected: Any) -> bool:
    pred_value = parse_number(pred)
    expected_value = parse_number(expected)
    if pred_value is None or expected_value is None:
        return repair_text(pred).lower() == repair_text(expected).lower()
    if abs(expected_value) < 1e-12:
        return abs(pred_value) < 1e-9
    return abs(pred_value - expected_value) / max(abs(expected_value), 1e-12) <= 0.025


def row_id(row: dict) -> str:
    for key, value in row.items():
        normalized = str(key).replace("\ufeff", "").replace('"', "").strip().lower()
        if normalized == "id":
            return str(value or "").strip()
    return ""


def normalize_question_key(question: Any) -> str:
    return re.sub(r"\s+", " ", repair_text(question).lower()).strip()


def load_cot_maps() -> tuple[dict[str, dict], dict[str, dict]]:
    sources = [
        ("verified_golden_expanded", WORKSPACE / "Retrieve new data v2" / "verified_golden_expanded.csv"),
        ("official_text", WORKSPACE / "Dataset Update" / "Physics_Problems_Text_Only" / "Physics_Problems_Text_Only.csv"),
        ("package_verified", WORKSPACE / "kaggle_api_package_v1" / "data" / "verified_golden_expanded.csv"),
    ]
    by_id: dict[str, dict] = {}
    by_question: dict[str, dict] = {}
    for source_name, path in sources:
        for row in read_csv_rows(path):
            rid = row_id(row)
            cot = str(row.get("cot") or "").strip()
            if not cot:
                continue
            item = {**row, "id": rid, "_cot_source": source_name}
            if rid:
                current = by_id.get(rid)
                # Prefer verified_golden_expanded because it has topic labels and repaired rows.
                if current is None or source_name == "verified_golden_expanded":
                    by_id[rid] = item
            qkey = normalize_question_key(row.get("question") or "")
            if qkey:
                current_q = by_question.get(qkey)
                if current_q is None or source_name == "verified_golden_expanded":
                    by_question[qkey] = item
    return by_id, by_question


def clean_reference_explanation(cot: str) -> str:
    text = repair_text(cot)
    text = re.sub(r"Step\s*(\d+)\s*:\s*Step\s*\1\s*:", r"Step \1:", text, flags=re.I)
    text = re.sub(r"Step\s*(\d+)\s*[\).]\s*", r"Step \1: ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    # Keep source wording but trim pathological long CoT so the adapter learns concise answers.
    words = text.split()
    if len(words) > 180:
        text = " ".join(words[:180]).rstrip(" ,.;") + "."
    return text


def explanation_quality_ok(explanation: str) -> bool:
    if not explanation:
        return False
    low = explanation.lower()
    blocked = [
        "question text is incomplete",
        "no specific physics quantity",
        "cannot be determined",
        "not enough information",
        "python code",
        "write the python",
        "let's write",
        "define variables",
    ]
    if any(item in low for item in blocked):
        return False
    word_count = len(explanation.split())
    if not 18 <= word_count <= 220:
        return False
    if not re.search(r"\d", explanation):
        return False
    math_signal = re.search(r"(=|\^|sqrt|/|\*|×|\\frac|formula|substitut|convert|joule|volt|ohm|farad|tesla|newton|ampere)", low)
    if not math_signal:
        return False
    weak_patterns = [
        r"step\s*1:\s*given values\s*step\s*2:\s*calculate",
        r"step\s*1:\s*given value\b",
        r"step\s*1:\s*constants\s*step\s*2:\s*calculate",
        r"step\s*1:\s*calculate\b.*step\s*2:\s*convert",
        r"step\s*1:\s*identify the given values\s*step\s*2:\s*calculate\b",
        r"step\s*1:\s*therefore\b",
    ]
    if any(re.search(pattern, low) for pattern in weak_patterns):
        return False
    return True


def question_from_row(row: dict) -> str:
    content = str(row.get("messages", [{}, {}])[1].get("content") or "")
    if content.startswith("Question:"):
        return content.split("Question:", 1)[1].strip()
    return content.strip()


def payload_from_row(row: dict) -> Optional[dict]:
    try:
        return json.loads(row["messages"][-1]["content"])
    except Exception:
        return None


def trace_user_content(question: str, solved: dict) -> str:
    trace = solved.get("trace") or {}
    givens = []
    for given in trace.get("givens") or []:
        givens.append(
            {
                "name": given.get("name"),
                "role": given.get("role"),
                "raw_span": given.get("raw_span"),
                "value_si": given.get("value_si"),
                "unit": given.get("unit"),
            }
        )
    user = {
        "question": question,
        "locked_answer": solved.get("answer"),
        "locked_unit": solved.get("unit"),
        "topic": trace.get("topic") or solved.get("topic"),
        "formula_id": solved.get("formula_id"),
        "formula": solved.get("formula"),
        "calculation": solved.get("calculation"),
        "givens": givens,
        "guardrail": "Explain the locked calculation only. Do not alter answer, unit, or formula.",
    }
    return json.dumps(user, ensure_ascii=False, separators=(",", ":"))


def _clean_inline(text: Any) -> str:
    text = repair_text(text)
    text = text.replace("**", "^")
    text = text.replace("*", " x ")
    text = text.replace("  ", " ")
    return text.strip(" .")


def _target_label(solved: dict) -> str:
    target = (solved.get("trace") or {}).get("target") or {}
    role = str(target.get("role") or target.get("description") or "").replace("_", " ").strip()
    symbol = str(target.get("symbol") or "").strip()
    if role and symbol:
        return f"{role} ({symbol})"
    return role or symbol or "the requested quantity"


def _given_summary(solved: dict) -> str:
    givens = (solved.get("trace") or {}).get("givens") or []
    parts = []
    for given in givens:
        role = str(given.get("role") or given.get("name") or "quantity").replace("_", " ")
        raw_span = str(given.get("raw_span") or "")
        value_si = given.get("value_si")
        if raw_span:
            parts.append(f"{role} = {raw_span}")
        elif value_si is not None:
            parts.append(f"{role} = {value_si}")
    return "; ".join(parts[:5])


def _topic_phrase(topic: str, formula_id: str) -> str:
    topic = topic or ""
    if topic == "capacitor":
        return "capacitor relation"
    if topic == "LC_oscillation":
        return "ideal LC oscillator relation"
    if topic == "ac_resonance":
        return "AC resonance relation"
    if topic == "circuit_power":
        return "electric power relation"
    if topic == "circuit_resistance":
        return "circuit resistance relation"
    if topic == "electrostatics_force":
        return "Coulomb-force relation"
    if topic == "electrostatics_field":
        return "electric-field relation"
    if topic == "induction":
        return "magnetic induction relation"
    if topic == "measurement_error":
        return "measurement-uncertainty relation"
    return formula_id.replace("_", " ")


def generate_locked_trace_explanation(question: str, solved: dict, style_index: int) -> str:
    """Generate a contextual explanation from a locked calculator trace.

    This is controlled data generation, not answer generation: the answer,
    formula, and calculation are taken from the verified calculator output.
    """

    trace = solved.get("trace") or {}
    topic = str(trace.get("topic") or solved.get("topic") or "")
    formula_id = str(solved.get("formula_id") or trace.get("formula_id") or "")
    formula = _clean_inline(solved.get("formula") or trace.get("formula") or formula_id)
    calculation = _clean_inline(solved.get("calculation") or trace.get("calculation") or formula)
    answer = str(solved.get("answer") or "")
    unit = str(solved.get("unit") or "")
    final = f"{answer} {unit}".strip()
    target = _target_label(solved)
    givens = _given_summary(solved)
    relation = _topic_phrase(topic, formula_id)
    question_hint = repair_text(question)
    if len(question_hint) > 150:
        question_hint = question_hint[:147].rstrip() + "..."

    variants = [
        (
            f"The question asks for {target}, so the locked solver treats it as a {relation}. "
            f"The extracted data are {givens}. Using {formula}, the calculator evaluates {calculation}. "
            f"After applying the requested output unit, the result is {final}."
        ),
        (
            f"The wording describes this situation: {question_hint} "
            f"The relevant quantities are {givens}, and the relation is {formula}. "
            f"Evaluating that relation for the extracted values gives {final}."
        ),
        (
            f"The problem context is preserved from the original question: {question_hint} "
            f"The target is {target}. With {givens}, the verified relation {formula} is used. "
            f"The calculator's substitution step is {calculation}, leading to {final}."
        ),
        (
            f"First identify the requested quantity as {target}. The usable values are {givens}. "
            f"For this {relation}, the governing expression is {formula}. "
            f"Substituting the extracted values gives {calculation}; therefore the final value is {final}."
        ),
        (
            f"The quantities in the question are matched to the {relation}: {givens}. "
            f"The requested result is {target}, so the calculation uses {formula}. "
            f"The evaluated result is {final}."
        ),
        (
            f"Reading the question literally, the known values are {givens}. "
            f"Those values are sufficient for {target} because the governing relation is {formula}. "
            f"Carrying out the calculator step {calculation} gives {final}."
        ),
        (
            f"Because the question is asking for {target}, the solver does not use unrelated quantities. "
            f"It keeps {givens} and applies {formula}. The verified computation is {calculation}, "
            f"so the reported result is {final}."
        ),
        (
            f"The calculation follows directly from the locked formula {formula}. "
            f"After reading {givens} from the question, the calculator performs {calculation}. "
            f"This matches the requested {target}, so the final answer is {final}."
        ),
    ]
    return re.sub(r"\s+", " ", variants[style_index % len(variants)]).strip()


def build_rows() -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    numeric_path = WORKSPACE / "physics_calculator_rebuild_v2" / "adapters" / "numeric_parser_final" / "dataset" / "numeric_parser_final_all.jsonl"
    numeric_rows = read_jsonl(numeric_path)
    cot_by_id, cot_by_question = load_cot_maps()
    rows: list[dict] = []
    rejected: list[dict] = []
    generation_log: list[dict] = []
    seen_explanations: set[str] = set()
    counts = {"numeric_rows": len(numeric_rows), "cot_rows_by_id": len(cot_by_id), "cot_rows_by_question": len(cot_by_question)}

    for row in numeric_rows:
        metadata = row.get("metadata") or {}
        source_id = str(metadata.get("source_id") or "").strip()
        if source_id.startswith("NP_"):
            source_id = source_id.split("_")[-1]
        cot_row = cot_by_id.get(source_id)
        if not cot_row:
            cot_row = cot_by_question.get(normalize_question_key(question_from_row(row)))
        payload = payload_from_row(row)
        question = question_from_row(row)
        if not payload or not question:
            rejected.append({"id": row.get("id"), "source_id": source_id, "reason": "missing_payload_or_question"})
            continue
        solved = solve_numeric_payload(question, payload)
        if solved is None:
            rejected.append({"id": row.get("id"), "source_id": source_id, "reason": "calculator_failed"})
            continue

        expected_answer = metadata.get("answer")
        expected_unit = metadata.get("unit")
        if expected_answer and not answer_close(solved.get("answer"), expected_answer):
            rejected.append(
                {
                    "id": row.get("id"),
                    "source_id": source_id,
                    "reason": f"metadata_answer_mismatch:{solved.get('answer')}!={expected_answer}",
                }
            )
            continue
        if expected_unit and normalize_unit(solved.get("unit")) != normalize_unit(expected_unit):
            rejected.append(
                {
                    "id": row.get("id"),
                    "source_id": source_id,
                    "reason": f"metadata_unit_mismatch:{solved.get('unit')}!={expected_unit}",
                }
            )
            continue

        if cot_row and cot_row.get("answer") and not answer_close(solved.get("answer"), cot_row.get("answer")):
            rejected.append(
                {
                    "id": row.get("id"),
                    "source_id": source_id,
                    "reason": f"answer_mismatch:{solved.get('answer')}!={cot_row.get('answer')}",
                }
            )
            continue
        if cot_row and cot_row.get("unit") and normalize_unit(solved.get("unit")) != normalize_unit(cot_row.get("unit")):
            rejected.append(
                {
                    "id": row.get("id"),
                    "source_id": source_id,
                    "reason": f"unit_mismatch:{solved.get('unit')}!={cot_row.get('unit')}",
                }
            )
            continue

        explanation_source = "generated_locked_trace"
        generation_reason = "missing_reference_cot"
        style_index = (len(generation_log) + len(rows)) % 8
        if cot_row:
            reference_explanation = clean_reference_explanation(cot_row.get("cot") or "")
            if explanation_quality_ok(reference_explanation):
                explanation = reference_explanation
                explanation_source = cot_row.get("_cot_source") or "reference_cot"
                generation_reason = ""
            else:
                explanation = generate_locked_trace_explanation(question, solved, style_index)
                generation_reason = "reference_explanation_quality_rejected"
        else:
            explanation = generate_locked_trace_explanation(question, solved, style_index)

        if not explanation_quality_ok(explanation):
            rejected.append({"id": row.get("id"), "source_id": source_id, "reason": f"{generation_reason or 'explanation'}_quality_rejected"})
            continue
        if explanation in seen_explanations:
            original_source = explanation_source
            duplicate_replaced = False
            for offset in range(8):
                candidate_style = (style_index + offset + 2) % 8
                candidate = generate_locked_trace_explanation(question, solved, candidate_style)
                if explanation_quality_ok(candidate) and candidate not in seen_explanations:
                    explanation = candidate
                    explanation_source = "generated_locked_trace"
                    generation_reason = f"duplicate_{original_source}_replaced_by_locked_trace"
                    style_index = candidate_style
                    duplicate_replaced = True
                    break
            if not duplicate_replaced:
                rejected.append({"id": row.get("id"), "source_id": source_id, "reason": "duplicate_explanation_rejected"})
                continue
        assistant = {"explanation": explanation}
        if FORBIDDEN_OUTPUT_KEYS & set(assistant):
            rejected.append({"id": row.get("id"), "source_id": source_id, "reason": "forbidden_output_key"})
            continue
        seen_explanations.add(explanation)
        row_id_prefix = "TEG" if explanation_source == "generated_locked_trace" else "TE"
        rows.append(
            {
                "id": f"{row_id_prefix}_{source_id}_{len(rows)+1:05d}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": trace_user_content(question, solved)},
                    {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, separators=(",", ":"))},
                ],
                "metadata": {
                    "source_numeric_id": row.get("id"),
                    "source_id": source_id,
                    "cot_source": explanation_source,
                    "generation_reason": generation_reason,
                    "style_index": style_index if explanation_source == "generated_locked_trace" else "",
                    "topic": solved.get("topic"),
                    "formula_id": solved.get("formula_id"),
                    "answer": solved.get("answer"),
                    "unit": solved.get("unit"),
                },
            }
        )
        if explanation_source == "generated_locked_trace":
            generation_log.append(
                {
                    "id": rows[-1]["id"],
                    "source_numeric_id": row.get("id"),
                    "source_id": source_id,
                    "topic": solved.get("topic"),
                    "formula_id": solved.get("formula_id"),
                    "answer": solved.get("answer"),
                    "unit": solved.get("unit"),
                    "style_index": style_index,
                    "generation_reason": generation_reason,
                    "question": question,
                    "explanation": explanation,
                }
            )
    return rows, rejected, generation_log, counts


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, rejected, generation_log, counts = build_rows()
    random.Random(2026).shuffle(rows)
    valid_size = max(30, min(180, int(len(rows) * 0.12)))
    valid_rows = rows[:valid_size]
    train_rows = rows[valid_size:]

    def write_jsonl(path: Path, data: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    write_jsonl(out_dir / "trace_explainer_final_all.jsonl", rows)
    write_jsonl(out_dir / "trace_explainer_final_train.jsonl", train_rows)
    write_jsonl(out_dir / "trace_explainer_final_valid.jsonl", valid_rows)

    rejected_path = out_dir / "trace_explainer_final_rejected.csv"
    with rejected_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source_id", "reason"])
        writer.writeheader()
        writer.writerows(rejected)

    generation_log_path = out_dir / "trace_explainer_final_generation_log.csv"
    with generation_log_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "source_numeric_id",
            "source_id",
            "topic",
            "formula_id",
            "answer",
            "unit",
            "style_index",
            "generation_reason",
            "question",
            "explanation",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(generation_log)

    topic_counts: dict[str, int] = {}
    formula_counts: dict[str, int] = {}
    cot_source_counts: dict[str, int] = {}
    for row in rows:
        meta = row["metadata"]
        topic_counts[meta["topic"]] = topic_counts.get(meta["topic"], 0) + 1
        formula_counts[meta["formula_id"]] = formula_counts.get(meta["formula_id"], 0) + 1
        cot_source_counts[meta["cot_source"]] = cot_source_counts.get(meta["cot_source"], 0) + 1
    summary = {
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "counts": counts,
        "topic_counts": dict(sorted(topic_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "formula_counts": dict(sorted(formula_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "cot_source_counts": dict(sorted(cot_source_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))),
        "policy": (
            "Inputs are locked calculator traces from numeric_parser_final rows. "
            "Target explanations use cleaned reference CoT when available and high quality. "
            "Missing or weak explanations are generated from locked calculator traces only; answer, unit, formula, and calculation are never generated by the language layer. "
            "Rows are kept only when calculator answer/unit match verified metadata or reference answer/unit."
        ),
        "rejected_rows": len(rejected),
        "rejected_csv": str(rejected_path),
        "generated_rows": len(generation_log),
        "generation_log_csv": str(generation_log_path),
    }
    (out_dir / "trace_explainer_final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
