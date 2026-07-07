from __future__ import annotations

import argparse
import ast
import builtins
import csv
import json
import math
import random
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "result"
DEFAULT_SOURCE = PROJECT_ROOT / "Retrieve new data v2" / "verified_golden_official_safe.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "physics_planner"

if str(RESULT_DIR) not in sys.path:
    sys.path.insert(0, str(RESULT_DIR))

import physics_engine_core as core  # noqa: E402


SYSTEM_PROMPT = (
    "You are a physics planner for educational physics problems. "
    "Return only valid JSON. Use SI units internally. "
    "The locked answer and locked unit are provided for supervision; your plan must be consistent with them. "
    "The python_code must be executable and must define final_result."
)

REQUIRED_PLANNER_KEYS = [
    "topic",
    "givens",
    "target",
    "formula",
    "python_code",
    "answer",
    "confidence",
]

TOPIC_FORMULAS = {
    "LC_oscillation": {
        "name": "LC oscillator relation",
        "expression": "Use LC frequency relations and/or energy conservation depending on the target.",
        "reason": "Ideal LC circuits conserve electromagnetic energy and satisfy omega = 1/sqrt(L*C).",
    },
    "capacitor": {
        "name": "Capacitor relations",
        "expression": "Q = C*U; W = 0.5*C*U**2; equivalent capacitance follows series/parallel rules.",
        "reason": "Capacitor problems connect charge, voltage, capacitance, energy, and equivalent capacitance.",
    },
    "electrostatics_field": {
        "name": "Electric field relation",
        "expression": "E = k*abs(q)/r**2, with vector summation when multiple fields are present.",
        "reason": "The field of point charges follows Coulomb's field law and fields superpose vectorially.",
    },
    "electrostatics_force": {
        "name": "Electric force relation",
        "expression": "F = k*abs(q1*q2)/r**2 or F = abs(q)*E, with vector summation when needed.",
        "reason": "Electric force follows Coulomb's law or the uniform-field relation.",
    },
    "ac_resonance": {
        "name": "AC resonance relation",
        "expression": "At resonance X_L = X_C and Z = R; f0 = 1/(2*pi*sqrt(L*C)).",
        "reason": "A series RLC circuit at resonance has canceling reactances.",
    },
    "circuit_resistance": {
        "name": "Circuit resistance relation",
        "expression": "R = U/I; R_series = sum(R_i); 1/R_parallel = sum(1/R_i); Z = sqrt(R**2 + (XL-XC)**2).",
        "reason": "Resistance and impedance problems use Ohm's law and connection rules.",
    },
    "circuit_power": {
        "name": "Circuit power relation",
        "expression": "P = U*I; P = I**2*R; P = U**2/R.",
        "reason": "Electric power is determined from the available voltage, current, and resistance data.",
    },
    "induction": {
        "name": "Electromagnetic induction relation",
        "expression": "Phi = B*S*cos(theta); abs(e) = N*abs(Delta Phi)/Delta t; W = 0.5*L*I**2.",
        "reason": "Induction problems use magnetic flux, Faraday's law, solenoid relations, or inductor energy.",
    },
    "measurement_error": {
        "name": "Measurement error relation",
        "expression": "absolute error, relative error, and propagated uncertainty follow the requested measurement convention.",
        "reason": "Measurement problems require consistent uncertainty definitions and units.",
    },
    "general_physics": {
        "name": "General physics relation",
        "expression": "Select the relevant vector, kinematics, or proportionality relation and compute consistently.",
        "reason": "General problems require identifying the requested relation from the problem statement.",
    },
}

UNIT_TARGET_SYMBOL = [
    ("A", "I", "current"),
    ("V", "U", "voltage or potential difference"),
    ("ohm", "R", "resistance or impedance"),
    ("W", "P", "power"),
    ("J", "W", "energy"),
    ("mJ", "W", "energy"),
    ("C", "Q", "charge"),
    ("uC", "Q", "charge"),
    ("nC", "Q", "charge"),
    ("F", "C", "capacitance"),
    ("uF", "C", "capacitance"),
    ("H", "L", "inductance"),
    ("mH", "L", "inductance"),
    ("Hz", "f", "frequency"),
    ("rad/s", "omega", "angular frequency"),
    ("s", "T", "time or period"),
    ("V/m", "E", "electric field strength"),
    ("N/C", "E", "electric field strength"),
    ("N", "F", "force"),
    ("T", "B", "magnetic field"),
    ("Wb", "Phi", "magnetic flux"),
    ("turns/m", "n", "turn density"),
    ("%", "relative_error", "relative error"),
    ("kg", "m", "mass"),
    ("m", "x", "length or distance"),
    ("cm", "x", "length or distance"),
]


@dataclass
class BuildResult:
    sample: dict[str, Any] | None
    reject_reason: str | None
    detail: str = ""


def ascii_unit(unit: Any) -> str:
    text = str(unit or "").strip()
    if text in {"", "-", "None", "none", "nan"}:
        return ""
    return core.canonical_unit(text).replace(" ", "")


def parse_numeric_answer(value: Any) -> float | None:
    try:
        return core.parse_number(str(value))
    except Exception:
        return None


def is_numeric_row(row: pd.Series) -> bool:
    if ascii_unit(row.get("unit")) == "":
        return False
    return parse_numeric_answer(row.get("answer")) is not None


def safe_import(name: str, globals_: dict[str, Any] | None = None, locals_: dict[str, Any] | None = None, fromlist=(), level: int = 0):
    if name == "math":
        return math
    if name == "numpy":
        return np
    if name == "np":
        return np
    raise ImportError(f"Import not allowed in planner code validation: {name}")


def execute_code_for_final_result(code: str) -> tuple[Any | None, str | None]:
    safe_builtins = {
        "__import__": safe_import,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "len": len,
        "range": range,
        "float": float,
        "int": int,
        "str": str,
        "pow": pow,
        "print": lambda *args, **kwargs: None,
    }
    env: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "math": math,
        "np": np,
        "numpy": np,
    }

    try:
        ast.parse(code)
        exec(code, env, env)
    except Exception:
        return None, traceback.format_exc(limit=2)

    if "final_result" not in env:
        return None, "python_code did not define final_result"

    final_result = env["final_result"]
    if isinstance(final_result, np.generic):
        final_result = final_result.item()
    return final_result, None


def sanitize_python_code(code: Any) -> str:
    text = str(code or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "final_result" not in text:
        return ""
    return text


def is_constant_answer_code(code: str) -> bool:
    text = str(code or "")
    if re.search(r"\banswer_value\s*=", text) or re.search(r"\banswer_text\s*=", text):
        return True

    compact = re.sub(r"\s+", "", text)
    if re.search(r"final_result=(?:answer_value|answer_text)\b", compact):
        return True

    meaningful_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("print("):
            continue
        if stripped in {"final_result", "final_result;"}:
            continue
        meaningful_lines.append(stripped)

    # Reject extremely short snippets that only assign a literal and final_result.
    if len(meaningful_lines) <= 2 and any("final_result" in line for line in meaningful_lines):
        joined = " ".join(meaningful_lines)
        if re.search(r"final_result\s*=\s*[-+]?[\d.]+(?:e[-+]?\d+)?\b", joined, re.I):
            return True

    return False


def infer_target(question: str, unit: str, topic: str) -> dict[str, str]:
    q = question.lower()
    unit_clean = ascii_unit(unit)

    if "maximum current" in q or "peak current" in q or "imax" in q:
        return {"symbol": "Imax", "unit": unit_clean, "description": "maximum current"}
    if "maximum voltage" in q or "peak voltage" in q or "umax" in q:
        return {"symbol": "Umax", "unit": unit_clean, "description": "maximum voltage"}

    for known_unit, symbol, desc in UNIT_TARGET_SYMBOL:
        if unit_clean == known_unit:
            return {"symbol": symbol, "unit": unit_clean, "description": desc}

    if "frequency" in q:
        return {"symbol": "f", "unit": unit_clean, "description": "frequency"}
    if "period" in q:
        return {"symbol": "T", "unit": unit_clean, "description": "period"}
    if "energy" in q:
        return {"symbol": "W", "unit": unit_clean, "description": "energy"}
    if "charge" in q:
        return {"symbol": "Q", "unit": unit_clean, "description": "charge"}
    if "capacitance" in q:
        return {"symbol": "C", "unit": unit_clean, "description": "capacitance"}
    if "inductance" in q:
        return {"symbol": "L", "unit": unit_clean, "description": "inductance"}
    if "electric field" in q or "field strength" in q:
        return {"symbol": "E", "unit": unit_clean, "description": "electric field strength"}
    if "force" in q:
        return {"symbol": "F", "unit": unit_clean, "description": "force"}
    if "resistance" in q:
        return {"symbol": "R", "unit": unit_clean, "description": "resistance"}
    if "current" in q:
        return {"symbol": "I", "unit": unit_clean, "description": "current"}
    if "voltage" in q or "potential difference" in q:
        return {"symbol": "U", "unit": unit_clean, "description": "voltage"}
    if "power" in q:
        return {"symbol": "P", "unit": unit_clean, "description": "power"}

    return {"symbol": "target", "unit": unit_clean, "description": f"target quantity for {topic}"}


def extract_givens(question: str) -> list[dict[str, Any]]:
    mentions = core.extract_quantity_mentions(question, max_items=12)
    givens: list[dict[str, Any]] = []
    seen: set[str] = set()

    for idx, raw in enumerate(mentions, start=1):
        raw_clean = str(raw).strip()
        if not raw_clean or raw_clean in seen:
            continue
        seen.add(raw_clean)

        match = re.match(rf"\s*({core.NUMBER_PATTERN})\s*(.+?)\s*$", core.clean_for_regex(raw_clean), flags=re.I)
        if not match:
            givens.append(
                {
                    "symbol": f"given_{idx}",
                    "value": None,
                    "unit": "",
                    "source": raw_clean,
                }
            )
            continue

        value = core.parse_number(match.group(1))
        unit = ascii_unit(match.group(2))
        si_value = None
        si_unit = unit
        try:
            if value is not None:
                si_value = value * core.unit_scale(unit)
                si_unit = si_unit_hint(unit)
        except Exception:
            pass

        givens.append(
            {
                "symbol": infer_given_symbol(raw_clean, unit, idx),
                "value": value,
                "unit": unit,
                "si_value": si_value,
                "si_unit": si_unit,
                "source": raw_clean,
            }
        )

    return givens


def si_unit_hint(unit: str) -> str:
    unit = ascii_unit(unit)
    mapping = {
        "uF": "F",
        "nF": "F",
        "pF": "F",
        "mF": "F",
        "uC": "C",
        "nC": "C",
        "pC": "C",
        "mC": "C",
        "mH": "H",
        "uH": "H",
        "mA": "A",
        "uA": "A",
        "mV": "V",
        "kV": "V",
        "cm": "m",
        "mm": "m",
        "mJ": "J",
        "uJ": "J",
        "nJ": "J",
    }
    return mapping.get(unit, unit)


def infer_given_symbol(raw: str, unit: str, index: int) -> str:
    low = raw.lower()
    unit = ascii_unit(unit)
    if unit in {"uF", "nF", "pF", "mF", "F"}:
        return "C" if index == 1 else f"C{index}"
    if unit in {"mH", "uH", "H"}:
        return "L" if index == 1 else f"L{index}"
    if unit in {"V", "mV", "kV"}:
        return "U" if "voltage" in low or "source" in low else f"U{index}"
    if unit in {"A", "mA", "uA"}:
        return "I" if index == 1 else f"I{index}"
    if unit in {"ohm"}:
        return "R" if index == 1 else f"R{index}"
    if unit in {"uC", "nC", "pC", "mC", "C"}:
        return "q" if index == 1 else f"q{index}"
    if unit in {"V/m", "N/C"}:
        return "E" if index == 1 else f"E{index}"
    if unit == "N":
        return "F" if index == 1 else f"F{index}"
    if unit in {"J", "mJ", "uJ", "nJ"}:
        return "W" if index == 1 else f"W{index}"
    if unit == "Hz":
        return "f" if index == 1 else f"f{index}"
    if unit == "rad/s":
        return "omega"
    if unit in {"s"}:
        return "t" if "time" in low else "T"
    if unit in {"cm", "mm", "m"}:
        return "r" if "distance" in low or "away" in low else f"x{index}"
    return f"given_{index}"


def formula_for_topic(topic: str, code: str, cot: str) -> dict[str, str]:
    formula = dict(TOPIC_FORMULAS.get(topic, TOPIC_FORMULAS["general_physics"]))
    formula["source"] = "topic_formula_bank"

    formula_lines: list[str] = []
    for line in str(cot or "").splitlines():
        if any(mark in line for mark in ["=", "√", "^", "pi", "π"]):
            cleaned = re.sub(r"^Step\s*\d+\s*:\s*", "", line).strip()
            if cleaned:
                formula_lines.append(cleaned)
        if len(formula_lines) >= 2:
            break

    if formula_lines:
        formula["expression"] = " ".join(formula_lines)
        formula["source"] = "official_cot"
    elif "final_result" in code:
        formula["source"] = "verified_golden_code"

    return formula


def build_user_prompt(row: pd.Series) -> str:
    return (
        f"Question: {str(row.get('question')).strip()}\n"
        f"Topic hint: {str(row.get('topic')).strip()}\n"
        "Return a JSON physics plan with givens, target, formula, executable python_code, answer, unit, and confidence. "
        "Do not assume the final answer is known; compute it in python_code and assign final_result."
    )


def build_planner_target(row: pd.Series, code: str) -> dict[str, Any]:
    topic = str(row.get("topic") or "general_physics").strip()
    question = str(row.get("question") or "").strip()
    unit = ascii_unit(row.get("unit"))
    cot = str(row.get("official_cot") or row.get("cot") or "")

    return {
        "topic": topic,
        "givens": extract_givens(question),
        "target": infer_target(question, unit, topic),
        "formula": formula_for_topic(topic, code, cot),
        "python_code": code,
        "answer": {
            "value": str(row.get("answer")).strip(),
            "unit": unit,
        },
        "confidence": 0.95,
    }


def validate_planner_target(target: dict[str, Any], locked_answer: Any, locked_unit: Any) -> tuple[bool, str]:
    for key in REQUIRED_PLANNER_KEYS:
        if key not in target:
            return False, f"missing key: {key}"

    if not isinstance(target.get("givens"), list):
        return False, "givens must be a list"
    if not isinstance(target.get("target"), dict):
        return False, "target must be an object"
    if not isinstance(target.get("formula"), dict):
        return False, "formula must be an object"
    if not isinstance(target.get("answer"), dict):
        return False, "answer must be an object"

    final_result, error = execute_code_for_final_result(str(target.get("python_code") or ""))
    if error:
        return False, error

    unit = target["answer"].get("unit") or locked_unit
    if not core.compare_answer(final_result, unit, locked_answer, locked_unit, rel_tol=5e-2, abs_tol=1e-6):
        return False, f"final_result {final_result!r} {unit} does not match locked answer {locked_answer!r} {locked_unit!r}"

    return True, "ok"


def row_to_sample(row: pd.Series) -> BuildResult:
    if not is_numeric_row(row):
        return BuildResult(None, "non_numeric_or_unitless")

    code = sanitize_python_code(row.get("golden_code"))
    if not code:
        return BuildResult(None, "missing_final_result_code")
    if is_constant_answer_code(code):
        return BuildResult(None, "constant_answer_code")

    final_result, error = execute_code_for_final_result(code)
    if error:
        return BuildResult(None, "golden_code_execution_failed", error)

    locked_unit = ascii_unit(row.get("unit"))
    if not core.compare_answer(final_result, locked_unit, row.get("answer"), locked_unit, rel_tol=5e-2, abs_tol=1e-6):
        return BuildResult(
            None,
            "golden_code_answer_mismatch",
            f"final_result={final_result!r}, answer={row.get('answer')!r}, unit={locked_unit!r}",
        )

    target = build_planner_target(row, code)
    valid, reason = validate_planner_target(target, row.get("answer"), locked_unit)
    if not valid:
        return BuildResult(None, "planner_target_validation_failed", reason)

    assistant = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    # Final parse check: train target must be strict JSON.
    try:
        json.loads(assistant)
    except Exception as exc:
        return BuildResult(None, "assistant_json_invalid", str(exc))

    sample = {
        "id": str(row.get("id")).strip(),
        "prefix": str(row.get("prefix")).strip(),
        "topic": str(row.get("topic")).strip(),
        "answer": str(row.get("answer")).strip(),
        "unit": locked_unit,
        "is_augmented": bool(row.get("is_augmented")) if "is_augmented" in row else None,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
            {"role": "assistant", "content": assistant},
        ],
    }
    return BuildResult(sample, None)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(args: argparse.Namespace) -> None:
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source)
    required = ["id", "prefix", "question", "golden_code", "answer", "unit", "topic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")

    if args.max_rows:
        df = df.head(args.max_rows).copy()

    samples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        result = row_to_sample(row)
        if result.sample is not None:
            samples.append(result.sample)
        else:
            rejected.append(
                {
                    "id": row.get("id", ""),
                    "prefix": row.get("prefix", ""),
                    "topic": row.get("topic", ""),
                    "answer": row.get("answer", ""),
                    "unit": row.get("unit", ""),
                    "reject_reason": result.reject_reason,
                    "detail": result.detail,
                    "question": row.get("question", ""),
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    valid_size = max(1, int(round(len(samples) * args.valid_ratio))) if samples else 0
    valid = samples[:valid_size]
    train = samples[valid_size:]

    write_jsonl(out_dir / "physics_planner_all.jsonl", samples)
    write_jsonl(out_dir / "physics_planner_train.jsonl", train)
    write_jsonl(out_dir / "physics_planner_valid.jsonl", valid)

    with (out_dir / "physics_planner_rejected.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "prefix", "topic", "answer", "unit", "reject_reason", "detail", "question"],
        )
        writer.writeheader()
        writer.writerows(rejected)

    summary = {
        "dataset_version": "v2_no_locked_answer_no_constant_code",
        "source": str(source),
        "rows_read": int(len(df)),
        "accepted": len(samples),
        "train": len(train),
        "valid": len(valid),
        "rejected": len(rejected),
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "accepted_by_topic": pd.Series([s["topic"] for s in samples]).value_counts().sort_index().to_dict() if samples else {},
        "accepted_by_prefix": pd.Series([s["prefix"] for s in samples]).value_counts().sort_index().to_dict() if samples else {},
        "rejected_by_reason": pd.Series([r["reject_reason"] for r in rejected]).value_counts().sort_index().to_dict() if rejected else {},
    }
    (out_dir / "physics_planner_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validated JSONL dataset for the physics planner adapter.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source verified CSV.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output folder.")
    parser.add_argument("--valid-ratio", type=float, default=0.08, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional limit for debugging.")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
