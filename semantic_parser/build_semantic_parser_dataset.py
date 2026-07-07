from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "result"
DEFAULT_SOURCE = PROJECT_ROOT / "Retrieve new data v2" / "verified_golden_official_safe.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "semantic_parser"

if str(RESULT_DIR) not in sys.path:
    sys.path.insert(0, str(RESULT_DIR))

import physics_engine_core as core  # noqa: E402


SYSTEM_PROMPT = (
    "You are a semantic parser for educational physics problems. "
    "Return only valid JSON. Do not solve the problem and do not output the final numeric answer. "
    "Your job is to normalize the natural-language question into a canonical physics task with topic, "
    "canonical_problem, target, givens, relations, constraints, and assumptions."
)

REQUIRED_KEYS = [
    "topic",
    "canonical_problem",
    "target",
    "givens",
    "relations",
    "constraints",
    "assumptions",
    "confidence",
]

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
    ("s", "T", "period or time"),
    ("V/m", "E", "electric field strength"),
    ("N/C", "E", "electric field strength"),
    ("N", "F", "force"),
    ("mN", "F", "force"),
    ("uN", "F", "force"),
    ("T", "B", "magnetic field"),
    ("Wb", "Phi", "magnetic flux"),
    ("%", "relative_error", "relative error"),
]

TOPIC_RELATIONS = {
    "LC_oscillation": ["omega = 1/sqrt(L*C)", "f = 1/(2*pi*sqrt(L*C))", "0.5*C*U^2 + 0.5*L*I^2 is conserved"],
    "capacitor": ["Q = C*U", "W = 0.5*C*U^2", "series and parallel equivalent capacitance rules"],
    "electrostatics_field": ["E = k*abs(q)/r^2", "electric fields superpose vectorially"],
    "electrostatics_force": ["F = k*abs(q1*q2)/r^2", "electric forces superpose vectorially"],
    "ac_resonance": ["X_L = X_C at resonance", "f0 = 1/(2*pi*sqrt(L*C))", "at resonance Z = R"],
    "circuit_resistance": ["U = I*R", "R_series = sum(R_i)", "1/R_parallel = sum(1/R_i)"],
    "circuit_power": ["P = U*I", "P = I^2*R", "P = U^2/R"],
    "induction": ["Phi = B*S*cos(theta)", "abs(e) = N*abs(Delta Phi)/Delta t", "W = 0.5*L*I^2", "B = mu0*n*I for a long solenoid"],
    "measurement_error": ["relative error = absolute error / measured value", "for products or quotients, relative errors add by the stated convention"],
    "general_physics": ["identify the required relation from the physical context", "use vector addition when directions matter"],
}


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
    return bool(ascii_unit(row.get("unit"))) and parse_numeric_answer(row.get("answer")) is not None


def si_unit_hint(unit: str) -> str:
    unit = ascii_unit(unit)
    mapping = {
        "uF": "F", "nF": "F", "pF": "F", "mF": "F",
        "uC": "C", "nC": "C", "pC": "C", "mC": "C",
        "mH": "H", "uH": "H",
        "mA": "A", "uA": "A",
        "mV": "V", "kV": "V",
        "cm": "m", "mm": "m",
        "cm^2": "m^2", "mm^2": "m^2",
        "mJ": "J", "uJ": "J", "nJ": "J",
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
        return "U" if any(x in low for x in ["voltage", "potential", "source", "maximum"]) else f"U{index}"
    if unit in {"A", "mA", "uA"}:
        return "I" if index == 1 else f"I{index}"
    if unit == "ohm":
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
    if unit == "T":
        return "B" if index == 1 else f"B{index}"
    if unit == "Wb":
        return "Phi" if index == 1 else f"Phi{index}"
    if unit == "s":
        return "t" if "time" in low else "T"
    if unit in {"cm", "mm", "m"}:
        return "r" if any(x in low for x in ["distance", "away", "radius", "separation"]) else f"x{index}"
    return f"given_{index}"


def extract_givens(question: str) -> list[dict[str, Any]]:
    mentions = core.extract_quantity_mentions(question, max_items=14)
    givens: list[dict[str, Any]] = []
    seen: set[str] = set()

    for idx, raw in enumerate(mentions, start=1):
        raw_clean = str(raw).strip()
        if not raw_clean or raw_clean in seen:
            continue
        seen.add(raw_clean)

        match = re.match(rf"\s*({core.NUMBER_PATTERN})\s*(.+?)\s*$", core.clean_for_regex(raw_clean), flags=re.I)
        if not match:
            givens.append({"symbol": f"given_{idx}", "value": None, "unit": "", "source": raw_clean})
            continue

        value = core.parse_number(match.group(1))
        unit = ascii_unit(match.group(2))
        si_value = None
        si_unit = si_unit_hint(unit)
        try:
            if value is not None:
                si_value = value * core.unit_scale(unit)
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


def infer_target(question: str, unit: str, topic: str) -> dict[str, str]:
    q = question.lower()
    unit_clean = ascii_unit(unit)

    if any(x in q for x in ["maximum current", "peak current", "imax", "i0"]):
        return {"symbol": "I0", "unit": unit_clean, "description": "maximum current"}
    if any(x in q for x in ["maximum voltage", "peak voltage", "umax", "u0"]):
        return {"symbol": "U0", "unit": unit_clean, "description": "maximum voltage"}

    for known_unit, symbol, desc in UNIT_TARGET_SYMBOL:
        if unit_clean == known_unit:
            return {"symbol": symbol, "unit": unit_clean, "description": desc}

    keywords = [
        ("angular frequency", "omega", "angular frequency"),
        ("frequency", "f", "frequency"),
        ("period", "T", "period"),
        ("energy", "W", "energy"),
        ("capacitance", "C", "capacitance"),
        ("inductance", "L", "inductance"),
        ("electric field", "E", "electric field strength"),
        ("field strength", "E", "electric field strength"),
        ("force", "F", "force"),
        ("charge", "Q", "charge"),
        ("resistance", "R", "resistance"),
        ("current", "I", "current"),
        ("voltage", "U", "voltage"),
        ("potential difference", "U", "voltage"),
        ("power", "P", "power"),
    ]
    for key, symbol, desc in keywords:
        if key in q:
            return {"symbol": symbol, "unit": unit_clean, "description": desc}
    return {"symbol": "target", "unit": unit_clean, "description": f"target quantity for {topic}"}


def has_unit(givens: list[dict[str, Any]], units: set[str]) -> bool:
    return any(ascii_unit(g.get("unit")) in units for g in givens)


def infer_canonical_problem(question: str, topic: str, target: dict[str, str], givens: list[dict[str, Any]]) -> str:
    q = question.lower()
    symbol = target.get("symbol", "")
    desc = target.get("description", "")

    has_c = has_unit(givens, {"F", "uF", "nF", "pF", "mF"})
    has_l = has_unit(givens, {"H", "mH", "uH"})
    has_u = has_unit(givens, {"V", "mV", "kV"})
    has_i = has_unit(givens, {"A", "mA", "uA"})
    has_w = has_unit(givens, {"J", "mJ", "uJ", "nJ"})
    has_r = has_unit(givens, {"ohm"})

    if topic == "LC_oscillation":
        if symbol == "omega":
            return "lc_angular_frequency_from_L_C"
        if symbol == "f":
            return "lc_frequency_from_L_C"
        if symbol == "I0" and has_c and has_l and has_u:
            return "lc_max_current_from_capacitor_voltage"
        if symbol == "U0" and has_c and has_l and has_i:
            return "lc_max_voltage_from_inductor_current"
        if "energy" in desc and has_c and has_u:
            return "lc_energy_from_capacitor_voltage"
        if "energy" in desc and has_l and has_i:
            return "lc_energy_from_inductor_current"
        return "lc_oscillation_general"

    if topic == "capacitor":
        if "parallel plate" in q or "parallel-plate" in q or "plates" in q or "separation" in q:
            return "parallel_plate_capacitor"
        if "series" in q or "parallel" in q or "equivalent" in q:
            return "capacitor_equivalent_capacitance"
        if symbol == "Q" and has_c and has_u:
            return "capacitor_charge_from_C_U"
        if symbol == "W" and has_c and has_u:
            return "capacitor_energy_from_C_U"
        if symbol == "C" and has_w and has_u:
            return "capacitor_capacitance_from_energy_voltage"
        if symbol == "U" and has_w and has_c:
            return "capacitor_voltage_from_energy_capacitance"
        if "distance" in q:
            return "parallel_plate_capacitor_change"
        return "capacitor_general"

    if topic == "ac_resonance":
        if symbol == "f":
            return "rlc_resonance_frequency_from_L_C"
        if symbol == "C":
            return "rlc_resonance_capacitance_from_L_f"
        if symbol == "L":
            return "rlc_resonance_inductance_from_C_f"
        if symbol == "P":
            return "rlc_resonance_power_from_I_R_or_U_R"
        return "ac_resonance_general"

    if topic in {"circuit_resistance", "circuit_power"}:
        if topic == "circuit_power" or symbol == "P":
            return "circuit_power_from_U_I_R"
        if "parallel" in q:
            return "circuit_parallel_equivalent_or_branch_current"
        if "series" in q:
            return "circuit_series_equivalent_or_voltage_division"
        if symbol == "I" and has_u and has_r:
            return "ohm_current_from_voltage_resistance"
        if symbol == "U" and has_i and has_r:
            return "ohm_voltage_from_current_resistance"
        if symbol == "R" and has_u and has_i:
            return "ohm_resistance_from_voltage_current"
        return "circuit_relation_general"

    if topic == "induction":
        if symbol == "B" and ("turns/m" in q or "solenoid" in q):
            return "solenoid_magnetic_field_from_turn_density_current"
        if symbol == "W" and has_l and has_i:
            return "inductor_energy_from_L_I"
        if symbol == "L" and has_w and has_i:
            return "inductance_from_energy_current"
        if symbol == "Phi":
            return "magnetic_flux_from_B_area_angle"
        if "emf" in q or "induced" in q:
            return "faraday_emf_from_flux_change"
        return "induction_general"

    if topic == "measurement_error":
        if symbol == "relative_error":
            return "measurement_relative_error"
        if symbol == "R":
            return "measurement_resistance_uncertainty_from_U_I"
        return "measurement_error_general"

    if topic == "electrostatics_field":
        if "ring" in q:
            return "electric_field_on_axis_of_charged_ring"
        if "triangle" in q or "vertices" in q:
            return "electric_field_vector_superposition_triangle"
        return "electric_field_point_charge_or_superposition"

    if topic == "electrostatics_force":
        if "triangle" in q or "vertices" in q or "right-angle" in q:
            return "electric_force_vector_superposition_geometry"
        return "coulomb_force_between_point_charges"

    if topic == "general_physics":
        if symbol == "F" and "force" in q:
            return "vector_force_resultant_or_coulomb_geometry"
        return "general_physics_relation"

    return f"{topic}_general"


def infer_constraints(question: str) -> list[str]:
    q = question.lower()
    constraints: list[str] = []
    if "maximum" in q or "peak" in q:
        constraints.append("use maximum or peak value where stated")
    if "rms" in q or "effective" in q:
        constraints.append("use RMS/effective values for AC quantities")
    if "series" in q:
        constraints.append("series connection")
    if "parallel" in q:
        constraints.append("parallel connection")
    if "equilateral" in q:
        constraints.append("equilateral triangle geometry")
    if "right-angle" in q or "right angle" in q:
        constraints.append("right-angle geometry")
    if "resonance" in q:
        constraints.append("resonance condition")
    return constraints


def infer_assumptions(topic: str, canonical_problem: str) -> list[str]:
    assumptions = ["convert all quantities to SI units before computation"]
    if topic == "LC_oscillation":
        assumptions.append("ideal LC circuit conserves electromagnetic energy")
    if topic == "capacitor":
        assumptions.append("ideal capacitor relations apply")
    if topic in {"electrostatics_field", "electrostatics_force"}:
        assumptions.append("point-charge model and Coulomb constant k = 9e9 in SI units unless otherwise stated")
    if topic == "ac_resonance":
        assumptions.append("ideal series RLC resonance unless the question states otherwise")
    if topic in {"circuit_resistance", "circuit_power"}:
        assumptions.append("ideal circuit elements")
    if topic == "induction":
        assumptions.append("use SI electromagnetic constants and idealized formulas as stated")
    if topic == "measurement_error":
        assumptions.append("use the measurement-error convention implied by the question")
    if "vector" in canonical_problem or "geometry" in canonical_problem:
        assumptions.append("resolve vector quantities using geometry before taking the magnitude")
    return assumptions


def build_semantic_target(row: pd.Series) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    topic = str(row.get("topic") or "general_physics").strip()
    unit = ascii_unit(row.get("unit"))
    target = infer_target(question, unit, topic)
    givens = extract_givens(question)
    canonical_problem = infer_canonical_problem(question, topic, target, givens)

    return {
        "topic": topic,
        "canonical_problem": canonical_problem,
        "target": target,
        "givens": givens,
        "relations": TOPIC_RELATIONS.get(topic, TOPIC_RELATIONS["general_physics"]),
        "constraints": infer_constraints(question),
        "assumptions": infer_assumptions(topic, canonical_problem),
        "confidence": 0.95,
    }


def build_user_prompt(row: pd.Series) -> str:
    return (
        f"Question: {str(row.get('question')).strip()}\n"
        f"Topic hint: {str(row.get('topic')).strip()}\n"
        f"Prefix hint: {str(row.get('prefix')).strip()}\n"
        "Return canonical JSON only. Do not calculate the final answer."
    )


def validate_target(target: dict[str, Any]) -> tuple[bool, str]:
    for key in REQUIRED_KEYS:
        if key not in target:
            return False, f"missing key: {key}"
    if not isinstance(target["target"], dict):
        return False, "target must be object"
    if not isinstance(target["givens"], list):
        return False, "givens must be list"
    if not target["canonical_problem"]:
        return False, "empty canonical_problem"
    if not target["topic"]:
        return False, "empty topic"
    if not target["target"].get("unit"):
        return False, "empty target unit"
    return True, "ok"


def row_to_sample(row: pd.Series) -> BuildResult:
    if not str(row.get("question") or "").strip():
        return BuildResult(None, "empty_question")
    if not is_numeric_row(row):
        return BuildResult(None, "non_numeric_or_unitless")

    target = build_semantic_target(row)
    valid, reason = validate_target(target)
    if not valid:
        return BuildResult(None, "semantic_target_validation_failed", reason)

    assistant = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    try:
        json.loads(assistant)
    except Exception as exc:
        return BuildResult(None, "assistant_json_invalid", str(exc))

    unit = ascii_unit(row.get("unit"))
    sample = {
        "id": str(row.get("id")).strip(),
        "prefix": str(row.get("prefix")).strip(),
        "topic": str(row.get("topic")).strip(),
        "canonical_problem": target["canonical_problem"],
        "answer": str(row.get("answer")).strip(),
        "unit": unit,
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
    required = ["id", "prefix", "question", "answer", "unit", "topic"]
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

    write_jsonl(out_dir / "physics_semantic_parser_all.jsonl", samples)
    write_jsonl(out_dir / "physics_semantic_parser_train.jsonl", train)
    write_jsonl(out_dir / "physics_semantic_parser_valid.jsonl", valid)

    with (out_dir / "physics_semantic_parser_rejected.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "prefix", "topic", "answer", "unit", "reject_reason", "detail", "question"],
        )
        writer.writeheader()
        writer.writerows(rejected)

    summary = {
        "dataset_version": "semantic_parser_v1_question_to_canonical_json_no_answer_no_code",
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
        "accepted_by_canonical_problem": pd.Series([s["canonical_problem"] for s in samples]).value_counts().sort_index().to_dict() if samples else {},
        "rejected_by_reason": pd.Series([r["reject_reason"] for r in rejected]).value_counts().sort_index().to_dict() if rejected else {},
    }
    (out_dir / "physics_semantic_parser_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    schema = {
        "assistant_json_required_keys": REQUIRED_KEYS,
        "purpose": "Parse natural-language physics questions into canonical machine-readable tasks. The model must not solve or output the final numeric answer.",
        "example_output_shape": {
            "topic": "LC_oscillation",
            "canonical_problem": "lc_max_current_from_capacitor_voltage",
            "target": {"symbol": "I0", "unit": "A", "description": "maximum current"},
            "givens": [{"symbol": "L", "value": 50, "unit": "mH", "si_value": 0.05, "si_unit": "H", "source": "50 mH"}],
            "relations": ["0.5*C*U^2 + 0.5*L*I^2 is conserved"],
            "constraints": ["use maximum or peak value where stated"],
            "assumptions": ["convert all quantities to SI units before computation"],
            "confidence": 0.95,
        },
    }
    (out_dir / "semantic_parser_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = (
        "# Physics Semantic Parser Dataset\n\n"
        "Task: train Qwen2.5 3B to parse a natural-language Physics Type 2 question into canonical JSON. "
        "This adapter is not allowed to solve the problem, emit Python code, or output the final answer. "
        "The downstream Python canonical solver will compute the answer.\n\n"
        "Files:\n"
        "- `physics_semantic_parser_train.jsonl`\n"
        "- `physics_semantic_parser_valid.jsonl`\n"
        "- `physics_semantic_parser_all.jsonl`\n"
        "- `physics_semantic_parser_summary.json`\n"
        "- `semantic_parser_schema.json`\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSONL data for the physics semantic parser adapter.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source verified CSV.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output folder.")
    parser.add_argument("--valid-ratio", type=float, default=0.08, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional limit for debugging.")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
