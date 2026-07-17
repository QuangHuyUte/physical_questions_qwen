"""Readable Physics inference pipeline for EXACTS 2026.

This is the public pipeline file used by the API, audit scripts, and Kaggle
evaluation. The large deterministic solver registry lives in
`physics_engine_core.py`; this file keeps the execution path clear:

Block 1. Configuration and core imports.
Block 2. Dataset loading, routing, and retrieval.
Block 3. Deterministic solver orchestration.
Block 4. Qwen fallback prompt, JSON repair, schema validation, and code check.
Block 5. Reasoning trace and response formatting.
Block 6. CSV evaluation and CLI.

The fallback LLM is disabled by default. When enabled, it can only contribute an
answer after its JSON payload passes deterministic schema checks and its Python
code computes a matching final_result.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import physics_engine_core as core

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================================
# Block 1. Configuration
# ============================================================================

PIPELINE_NAME = "physics_readable_pipeline"
PIPELINE_VERSION = "clean_blocks_with_verified_llm_json_guardrail"

USE_LLM_FALLBACK = False
DEFAULT_LLM_MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
model = None
tokenizer = None
LLM_MODEL_INFO = {
    "enabled": False,
    "ready": False,
    "model_name_or_path": None,
    "load_in_4bit": None,
    "local_files_only": None,
    "error": None,
}

PIPELINE_STAGES = [
    "normalize_input",
    "route_topic_prefix",
    "build_candidate_topics",
    "retrieve_context_examples",
    "run_deterministic_solver",
    "optional_formula_planner",
    "optional_verified_llm_fallback",
    "build_reasoning_trace",
    "format_response",
]

PREFIX_TOPIC_FALLBACKS = {
    "LD": ["general_physics", "electrostatics_force", "electrostatics_field"],
    "DT": ["electrostatics_field"],
    "TD": ["capacitor"],
    "NL": ["capacitor", "induction", "LC_oscillation"],
    "DDT": ["induction", "LC_oscillation", "circuit_power", "circuit_resistance", "ac_resonance"],
    "CH": ["ac_resonance", "circuit_power", "circuit_resistance"],
    "THCB": ["measurement_error", "circuit_power", "circuit_resistance"],
}

LLM_REQUIRED_KEYS = ["givens", "target", "formula", "python_code", "answer", "explanation", "premises"]

LLM_JSON_TEMPLATE = {
    "givens": [
        {
            "symbol": "L",
            "value": 0.05,
            "unit": "H",
            "source": "50 mH inductor",
        }
    ],
    "target": {
        "symbol": "I_max",
        "unit": "A",
        "description": "maximum current in the LC circuit",
    },
    "formula": {
        "name": "LC energy conservation",
        "expression": "0.5*C*V_max**2 = 0.5*L*I_max**2",
        "reason": "At maximum capacitor voltage all energy is electric; at maximum current all energy is magnetic.",
    },
    "python_code": "L = 0.05\nC = 20e-6\nVmax = 12\nfinal_result = Vmax * math.sqrt(C / L)",
    "answer": {
        "value": 0.24,
        "unit": "A",
    },
    "explanation": "Use conservation of energy in the LC oscillator, convert mH and microfarad to SI, then solve for maximum current.",
    "premises": [
        "Ideal LC circuit, so total electromagnetic energy is conserved.",
        "Use SI units before substitution.",
    ],
}


# Public aliases used by existing helper scripts.
compare_answer = core.compare_answer
canonical_unit = core.canonical_unit
find_file = core.find_file
normalize_text = core.normalize_text
retrieve_examples = core.retrieve_examples
result = core.result


# ============================================================================
# Block 2. Dataset Loading, Routing, And Retrieval
# ============================================================================

def prepare_pipeline(verified_path: str | Path | None = None, verbose: bool = False) -> bool:
    """Load verified data, train lightweight routers, and prepare retrieval."""
    ok = core.prepare_pipeline(verified_path, verbose=verbose)
    return bool(ok)


def _ensure_prepared() -> None:
    if not hasattr(core, "topic_router") or not hasattr(core, "prefix_router"):
        prepare_pipeline()


def _route_question(question_norm: str) -> dict[str, Any]:
    _ensure_prepared()
    topic_arr, topic_conf_arr = core.predict_with_confidence(core.topic_router, [question_norm])
    prefix_arr, prefix_conf_arr = core.predict_with_confidence(core.prefix_router, [question_norm])
    return {
        "topic": topic_arr[0],
        "topic_conf": float(topic_conf_arr[0]),
        "prefix": prefix_arr[0],
        "prefix_conf": float(prefix_conf_arr[0]),
    }


def _candidate_topics(topic: str, prefix: str, question: str = "") -> list[str]:
    candidates = [topic]
    for extra in PREFIX_TOPIC_FALLBACKS.get(prefix, []):
        if extra not in candidates:
            candidates.append(extra)
    q = normalize_text(question).lower()
    heuristic_topics = []
    if re.search(r"\blc\b|lossless lc|ideal lc|oscillat", q):
        heuristic_topics.append("LC_oscillation")
    if "resonance" in q or "resonant" in q or "resonates" in q:
        heuristic_topics.append("ac_resonance")
    if "power" in q or "dissipated" in q:
        heuristic_topics.append("circuit_power")
    if "resistance" in q or "ohm" in q or "u/i" in q:
        heuristic_topics.append("circuit_resistance")
    for extra in heuristic_topics:
        if extra not in candidates:
            candidates.append(extra)
    return candidates


def _context_examples(question: str, topic: str, prefix: str, k: int = 4) -> list[dict[str, Any]]:
    return core.retrieve_examples(question, topic=topic, prefix=prefix, k=k)


# ============================================================================
# Block 3. Deterministic Solver Orchestration
# ============================================================================

def _run_deterministic_solver(question: str, candidate_topics: list[str]):
    for topic in candidate_topics:
        solver = core.SOLVERS.get(topic)
        if solver is None:
            continue
        sol = solver(question)
        if sol is not None:
            return sol, topic
    return None, None


def _run_formula_planner(question: str, candidate_topics: list[str]):
    sol = core.formula_planner_solve(question, candidate_topics)
    if sol is None:
        return None, None
    return sol, "formula_planner"


def _should_try_formula_first(question: str, route: dict[str, Any]) -> bool:
    """Prefer formula planning for explicit numeric targets that concept rules can misread."""
    q = normalize_text(question).lower()
    if route["topic"] == "LC_oscillation":
        asks_current = re.search(r"\b(maximum|max|peak|amplitude)\s+current\b|\bcurrent\s+(amplitude|max|maximum)\b", q)
        has_lc_data = re.search(r"\b(inductor|inductance|mh| h)\b", q) and re.search(r"\b(capacitor|capacitance|uf|microfarad| f)\b", q)
        has_voltage_or_energy = re.search(r"\bvoltage|potential|energy\b", q)
        if asks_current and has_lc_data and has_voltage_or_energy:
            return True
    return False


def _unanswered_solution(topic: str):
    return core.result(
        answer="",
        unit="-",
        explanation=(
            "No deterministic solver matched confidently. "
            "No answer is returned because USE_LLM_FALLBACK=False. "
            "Retrieved examples are provided only for debugging."
        ),
        topic=topic,
        confidence=0.0,
        method="unanswered_no_fallback",
        code="",
        cot=[
            "Step 1: Predict the physics topic and retrieve similar examples for context.",
            "Step 2: Try deterministic solvers for the predicted topic and prefix-aware fallback topics.",
            "Step 3: Since no solver matches confidently, return no numeric answer instead of guessing.",
        ],
        premises=[],
    )


# ============================================================================
# Block 4. Qwen Fallback JSON Guardrail
# ============================================================================

def _formula_candidates_for_topic(topic: str) -> list[dict[str, Any]]:
    bank = getattr(core, "FORMULA_PLANNER_BANK", [])
    candidates = []
    for item in bank:
        if item.get("topic") != topic:
            continue
        candidates.append({
            "name": item.get("name", ""),
            "topic": item.get("topic", ""),
            "requires": item.get("requires", []),
            "formula": item.get("formula", ""),
            "unit": item.get("unit", "dynamic") if isinstance(item.get("unit"), str) else "dynamic",
        })
    return candidates[:12]


def build_llm_fallback_prompt(question: str, topic: str, prefix: str, examples: list[dict[str, Any]]) -> str:
    """Build a strict fill-in JSON prompt for Qwen fallback."""
    example_cards = []
    for ex in examples[:4]:
        example_cards.append({
            "id": ex.get("id", ""),
            "prefix": ex.get("prefix", ""),
            "topic": ex.get("topic", ""),
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "unit": ex.get("unit", ""),
            "cot": str(ex.get("cot", ""))[:1000],
        })

    prompt = {
        "role": "physics_json_solver",
        "task": "Solve the question using physics reasoning and return one JSON object only.",
        "hard_rules": [
            "Return valid JSON only. Do not use Markdown fences.",
            "Fill exactly the template keys: givens, target, formula, python_code, answer, explanation, premises.",
            "python_code must be safe numeric Python and must set final_result.",
            "Do not import packages. math, numpy as np, pi, sqrt, K, EPS0, and MU0 are already available.",
            "Use SI units internally, then output the requested answer unit.",
            "The answer.value must numerically match final_result.",
            "Retrieved examples are style/context only. Do not copy their answers unless independently computed.",
        ],
        "predicted_topic": topic,
        "predicted_prefix": prefix,
        "question": question,
        "topic_formula_card": getattr(core, "FORMULA_BY_TOPIC", {}).get(topic, {}),
        "formula_bank_candidates": _formula_candidates_for_topic(topic),
        "retrieved_examples": example_cards,
        "json_template_to_fill": LLM_JSON_TEMPLATE,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def _strip_json_fences(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def repair_json_text(text: str) -> str:
    """Repair common Qwen JSON formatting mistakes before strict validation."""
    text = _strip_json_fences(text)
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
        "True": "true",
        "False": "false",
        "None": "null",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text.strip()


def extract_json_object(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    text = repair_json_text(str(payload))
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(text)
        except Exception as exc:
            raise ValueError(f"invalid JSON payload: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("LLM payload must be a JSON object")
    return obj


def _normalize_llm_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Accept a few old flat-output variants, then normalize to the template."""
    out = dict(data)
    if isinstance(out.get("answer"), (str, int, float)):
        out["answer"] = {"value": out.get("answer"), "unit": out.get("unit", "")}
    if isinstance(out.get("formula"), str):
        out["formula"] = {"name": "", "expression": out.get("formula", ""), "reason": ""}
    if isinstance(out.get("target"), str):
        out["target"] = {"symbol": "", "unit": out.get("unit", ""), "description": out.get("target", "")}
    if "givens" not in out and "given" in out:
        out["givens"] = out["given"]
    if "premises" not in out:
        out["premises"] = []
    return out


def _validate_llm_payload(data: dict[str, Any]) -> list[str]:
    errors = []
    missing = [key for key in LLM_REQUIRED_KEYS if key not in data]
    if missing:
        errors.append(f"missing required keys: {missing}")

    givens = data.get("givens")
    if not isinstance(givens, list) or not givens:
        errors.append("givens must be a non-empty list")
    else:
        for idx, item in enumerate(givens[:20]):
            if not isinstance(item, dict):
                errors.append(f"givens[{idx}] must be an object")
                continue
            for key in ["symbol", "value", "unit", "source"]:
                if key not in item:
                    errors.append(f"givens[{idx}] missing {key}")

    target = data.get("target")
    if not isinstance(target, dict):
        errors.append("target must be an object")
    else:
        for key in ["symbol", "unit", "description"]:
            if key not in target:
                errors.append(f"target missing {key}")

    formula = data.get("formula")
    if not isinstance(formula, dict):
        errors.append("formula must be an object")
    else:
        for key in ["name", "expression", "reason"]:
            if key not in formula:
                errors.append(f"formula missing {key}")

    answer = data.get("answer")
    if not isinstance(answer, dict):
        errors.append("answer must be an object")
    else:
        if "value" not in answer:
            errors.append("answer missing value")
        if "unit" not in answer:
            errors.append("answer missing unit")

    code = data.get("python_code")
    if not isinstance(code, str) or not code.strip():
        errors.append("python_code must be a non-empty string")
    elif "final_result" not in code:
        errors.append("python_code must assign final_result")
    elif re.search(r"\b(import|open|eval|exec|compile|input|__import__)\b", code):
        errors.append("python_code contains a forbidden operation")

    explanation = data.get("explanation")
    if not isinstance(explanation, str) or len(explanation.split()) < 10:
        errors.append("explanation must be an English physics explanation with at least 10 words")

    premises = data.get("premises")
    if not isinstance(premises, list):
        errors.append("premises must be a list")
    return errors


def _answer_value_as_text(value: Any) -> str:
    if isinstance(value, (int, float, np.floating)):
        if math.isfinite(float(value)):
            return f"{float(value):.6g}"
    return str(value).strip()


def verify_llm_fallback_payload(payload: Any, question: str, topic: str):
    """Parse, repair, validate, execute, and verify one Qwen fallback payload."""
    try:
        data = _normalize_llm_payload(extract_json_object(payload))
    except Exception as exc:
        return None, str(exc)

    errors = _validate_llm_payload(data)
    if errors:
        return None, "; ".join(errors)

    final_result, err = core.execute_fallback_code(data["python_code"])
    if err:
        return None, err

    answer_obj = data["answer"]
    unit = canonical_unit(answer_obj.get("unit") or data.get("unit") or data.get("target", {}).get("unit") or "-")
    answer_text = _answer_value_as_text(answer_obj.get("value"))

    if isinstance(final_result, (int, float, np.floating)):
        code_answer = f"{float(final_result):.6g}"
        if not compare_answer(answer_text, unit, code_answer, unit, rel_tol=5e-2):
            return None, f"answer {answer_text} {unit} does not match final_result {code_answer} {unit}"
        answer_text = code_answer

    formula = data.get("formula", {})
    premises = [str(x).strip() for x in data.get("premises", []) if str(x).strip()]
    if formula.get("expression"):
        premises.insert(0, "Formula: " + str(formula["expression"]))

    return core.result(
        answer=answer_text,
        unit=unit,
        explanation=str(data["explanation"]).strip(),
        topic=topic,
        confidence=0.62,
        method="llm_fallback_verified_json",
        code=data["python_code"],
        premises=premises[:8],
    ), None


def call_llm_fallback(prompt: str):
    """Call a locally attached Qwen model/callable. No remote API is used here."""
    if callable(model):
        return model(prompt)
    if model is None or tokenizer is None:
        return None

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted], return_tensors="pt")
    try:
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
    except Exception:
        pass
    outputs = model.generate(
        **inputs,
        max_new_tokens=900,
        temperature=0.0,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


def load_llm_fallback_model(
    model_name_or_path: str | Path | None = None,
    *,
    load_in_4bit: bool = True,
    local_files_only: bool = True,
    device_map: str = "auto",
):
    """Load an open-source <=8B Qwen fallback model for local API serving.

    `local_files_only=True` is the deployment-safe default: download/cache the
    model before starting the API, then serve without internet dependency.
    """
    global model, tokenizer, USE_LLM_FALLBACK, LLM_MODEL_INFO

    model_ref = str(model_name_or_path or os.environ.get("PHYSICS_LLM_MODEL_PATH") or DEFAULT_LLM_MODEL_NAME)
    LLM_MODEL_INFO = {
        "enabled": True,
        "ready": False,
        "model_name_or_path": model_ref,
        "load_in_4bit": bool(load_in_4bit),
        "local_files_only": bool(local_files_only),
        "error": None,
    }

    try:
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_ref,
                max_seq_length=4096,
                dtype=None,
                load_in_4bit=bool(load_in_4bit),
            )
            FastLanguageModel.for_inference(model)
        except ImportError:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_ref,
                trust_remote_code=True,
                local_files_only=bool(local_files_only),
            )
            kwargs = {
                "trust_remote_code": True,
                "local_files_only": bool(local_files_only),
                "device_map": device_map,
            }
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            else:
                kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
            model.eval()

        USE_LLM_FALLBACK = True
        LLM_MODEL_INFO["ready"] = True
        return True
    except Exception as exc:
        model = None
        tokenizer = None
        USE_LLM_FALLBACK = False
        LLM_MODEL_INFO["error"] = str(exc)
        raise


def llm_fallback_solve(question: str, topic: str, prefix: str, examples: list[dict[str, Any]]):
    prompt = build_llm_fallback_prompt(question, topic, prefix, examples)
    raw = call_llm_fallback(prompt)
    if raw is None:
        return None
    sol, err = verify_llm_fallback_payload(raw, question, topic)
    if sol is None:
        return None
    return sol


# ============================================================================
# Block 5. Reasoning Trace And Response Formatting
# ============================================================================

SUBSCRIPT_DIGITS = str.maketrans({
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
})

DISPLAY_FORMULA_OVERRIDES = {
    "formula_planner_lc_max_current": {
        "formulas": [
            "½CU₀² = ½LI₀²",
            "I₀ = U₀√(C/L)",
        ],
        "formula_latex": [
            r"\frac{1}{2} C U_0^2 = \frac{1}{2} L I_0^2",
            r"I_0 = U_0 \sqrt{\frac{C}{L}}",
        ],
        "reason": "Use conservation of energy in an ideal LC oscillator: maximum electric energy in the capacitor becomes maximum magnetic energy in the inductor.",
    },
}


def _display_math_text(text: Any) -> Any:
    """Make display-only math text cleaner without changing computation."""
    if not isinstance(text, str):
        return text
    s = text
    s = re.sub(r"\bpi\b", "π", s)
    s = re.sub(r"\bsqrt\s*\(", "√(", s)
    s = s.replace("1/2", "½").replace("1 / 2", "½")
    s = s.replace("^2", "²").replace("^3", "³")
    s = s.replace("U0", "U₀").replace("I0", "I₀")
    s = s.replace("Umax", "Uₘₐₓ").replace("Imax", "Iₘₐₓ")
    s = re.sub(r"(\d)\s+π", r"\1π", s)
    s = re.sub(r"\s+\*\s+", " × ", s)
    s = re.sub(r"(?<=\d)\*(?=√|\d)", "×", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _display_math_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: value if key == "formula_latex" else _display_math_obj(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_display_math_obj(value) for value in obj]
    return _display_math_text(obj)


def _formula_latex_from_text(formula: str) -> str:
    """Small display helper for common formulas; not used for solving."""
    f = str(formula)
    replacements = [
        (r"T = 2 pi sqrt(LC), f = 1/(2 pi sqrt(LC))", r"T = 2\pi\sqrt{LC},\quad f = \frac{1}{2\pi\sqrt{LC}}"),
        (r"W = 1/2 C U^2 + 1/2 L I^2", r"W = \frac{1}{2}CU^2 + \frac{1}{2}LI^2"),
        (r"1/2 C U0^2 = 1/2 L I0^2", r"\frac{1}{2}CU_0^2 = \frac{1}{2}LI_0^2"),
        (r"I/Imax = sqrt(W_magnetic/W_total)", r"\frac{I}{I_{\max}} = \sqrt{\frac{W_{\mathrm{magnetic}}}{W_{\mathrm{total}}}}"),
    ]
    for raw, latex in replacements:
        if raw == f:
            return latex
    f = f.replace("sqrt(", r"\sqrt{")
    f = f.replace("pi", r"\pi")
    f = f.replace("^2", "^2").replace("U0", "U_0").replace("I0", "I_0")
    return f


def _apply_display_formula_overrides(sol):
    method = getattr(sol, "method", "")
    override = DISPLAY_FORMULA_OVERRIDES.get(method)

    if override:
        sol.explanation = re.sub(
            r"The solver uses .*?\. Substitution and computation:",
            "The solver uses energy conservation: "
            + override["formulas"][0]
            + ", so "
            + override["formulas"][1]
            + ". Substitution and computation:",
            str(sol.explanation),
        )
        sol.cot = [
            (
                "Step 4: Apply energy conservation in the LC oscillator: "
                + override["formulas"][0]
                + ", hence "
                + override["formulas"][1]
                + "."
                if str(step).startswith("Step 4:")
                else step
            )
            for step in (sol.cot or [])
        ]
        sol.premises = [
            p for p in (sol.premises or [])
            if not str(p).startswith("Formula: T =")
        ]
        sol.premises = [
            "Selection: " + override["reason"]
            if str(p).startswith("Selection:")
            else p
            for p in sol.premises
        ]
        for formula in reversed(override["formulas"]):
            premise = "Formula: " + formula
            if premise not in sol.premises:
                sol.premises.insert(1 if sol.premises else 0, premise)

        trace = dict(getattr(sol, "trace", {}) or {})
        if trace:
            trace["formulas"] = override["formulas"]
            trace["formula_latex"] = override["formula_latex"]
            qa = dict(trace.get("question_analysis", {}) or {})
            qa["formula_selection_reason"] = override["reason"]
            trace["question_analysis"] = qa
            trace["proof_path"] = [
                item if item.get("stage") != "formula" else {
                    "stage": "formula",
                    "content": "; ".join(override["formulas"]),
                }
                for item in (trace.get("proof_path", []) or [])
            ]
            sol.trace = trace

    sol.explanation = _display_math_text(sol.explanation)
    sol.cot = [_display_math_text(step) for step in (sol.cot or [])]
    sol.premises = [_display_math_text(premise) for premise in (sol.premises or [])]
    if hasattr(sol, "trace"):
        trace = _display_math_obj(getattr(sol, "trace", {}) or {})
        if trace and "formula_latex" not in trace and trace.get("formulas"):
            trace["formula_latex"] = [_formula_latex_from_text(f) for f in trace.get("formulas", [])]
        sol.trace = trace
    return sol


def _enrich_reasoning(question: str, sol):
    return _apply_display_formula_overrides(core.enrich_solver_reasoning(question, sol))



def _reasoning_quality(out: dict[str, Any]) -> float:
    return core.reasoning_quality_score(out)


def _format_output(question: str, sol, route: dict[str, Any], examples: list[dict[str, Any]], verified: bool | None):
    trace = dict(getattr(sol, "trace", {}) or {})
    if trace:
        trace["version"] = PIPELINE_VERSION
    out = {
        "answer": sol.answer,
        "unit": sol.unit,
        "explanation": sol.explanation,
        "cot": sol.cot,
        "premises": sol.premises,
        "trace": trace,
        "confidence": sol.confidence,
        "topic_pred": route["topic"],
        "topic_conf": route["topic_conf"],
        "prefix_pred": route["prefix"],
        "prefix_conf": route["prefix_conf"],
        "solver_conf": sol.confidence,
        "method": sol.method,
        "verified_if_known": verified,
        "retrieved_ids": [ex.get("id", "") for ex in examples],
        "python_code": sol.code,
        "pipeline_name": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_stages": PIPELINE_STAGES,
    }
    out["reasoning_quality"] = _reasoning_quality(out)
    return out


def solve_physics_question(question: str, known_answer: Any = None, known_unit: Any = None) -> dict[str, Any]:
    """Main public entrypoint for a single Physics question."""
    question_norm = normalize_text(question)
    route = _route_question(question_norm)
    candidates = _candidate_topics(route["topic"], route["prefix"], question_norm)
    examples = _context_examples(question, route["topic"], route["prefix"])

    if _should_try_formula_first(question, route):
        sol, _ = _run_formula_planner(question, candidates)
        if sol is None:
            sol, _ = _run_deterministic_solver(question, candidates)
    else:
        sol, _ = _run_deterministic_solver(question, candidates)
        if sol is None:
            sol, _ = _run_formula_planner(question, candidates)

    verified = None
    if sol is not None and known_answer is not None:
        verified = compare_answer(sol.answer, sol.unit, known_answer, known_unit)

    if sol is None and USE_LLM_FALLBACK:
        sol = llm_fallback_solve(question, route["topic"], route["prefix"], examples)
        verified = compare_answer(sol.answer, sol.unit, known_answer, known_unit) if sol is not None and known_answer is not None else None

    if sol is None:
        sol = _unanswered_solution(route["topic"])
        verified = False if known_answer is not None else None

    sol = _enrich_reasoning(question, sol)
    return _format_output(question, sol, route, examples, verified)


def answer_physics_api(question: str, debug: bool = False) -> dict[str, Any]:
    """Small API response by default, full trace only when debug=True."""
    out = solve_physics_question(question)
    response = {
        "answer": out["answer"],
        "unit": out["unit"],
        "explanation": out["explanation"],
        "cot": out.get("cot", []),
        "premises": out.get("premises", []),
        "confidence": out.get("confidence", out.get("solver_conf", 0.0)),
        "pipeline_version": PIPELINE_VERSION,
    }
    if debug:
        response["reasoning_trace"] = out.get("trace", {})
        response["debug"] = {
            "topic_pred": out["topic_pred"],
            "topic_conf": out["topic_conf"],
            "prefix_pred": out["prefix_pred"],
            "prefix_conf": out["prefix_conf"],
            "method": out["method"],
            "reasoning_quality": out.get("reasoning_quality"),
            "retrieved_ids": out["retrieved_ids"],
            "pipeline_stages": PIPELINE_STAGES,
        }
    return response


# ============================================================================
# Block 6. CSV Evaluation And CLI
# ============================================================================

def evaluate_csv(input_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    data = pd.read_csv(input_path, dtype=str).fillna("")
    rows = []
    for _, row in data.iterrows():
        out = solve_physics_question(row["question"], row.get("answer"), row.get("unit"))
        rows.append({
            "id": row.get("id", ""),
            "question": row.get("question", ""),
            "true_answer": row.get("answer", ""),
            "true_unit": row.get("unit", ""),
            **out,
            "cot_steps": len(out.get("cot", []) or []),
            "premise_count": len(out.get("premises", []) or []),
            "is_correct": compare_answer(out["answer"], out["unit"], row.get("answer", ""), row.get("unit", "")) if "answer" in row else None,
            "attempted": out["method"] != "unanswered_no_fallback",
        })
    out_df = pd.DataFrame(rows)
    if output_path:
        out_df.to_csv(output_path, index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EXACTS Physics pipeline.")
    parser.add_argument("--verified", default=None, help="Path to verified_golden_expanded.csv")
    parser.add_argument("--input", default=None, help="CSV with question column for batch evaluation")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    parser.add_argument("--question", default=None, help="Single question to solve")
    parser.add_argument("--debug", action="store_true", help="Print debug fields")
    args = parser.parse_args()

    prepare_pipeline(args.verified, verbose=args.debug)

    if args.question:
        print(json.dumps(answer_physics_api(args.question, debug=args.debug), ensure_ascii=False, indent=2))
        return

    if args.input:
        result_df = evaluate_csv(args.input, args.output)
        if "is_correct" in result_df:
            print("accuracy", int(result_df["is_correct"].sum()), "/", len(result_df), float(result_df["is_correct"].mean()))
        if args.output:
            print("wrote", args.output)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
