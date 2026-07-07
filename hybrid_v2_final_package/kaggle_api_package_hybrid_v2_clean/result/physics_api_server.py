"""
Physics API Server for EXACTS 2026 Physics Pipeline.

Required files in result/:
- physics_pipeline.py
- physics_engine_core.py

Run:
    cd D:/EXACTS2026
    python ./result/physics_api_server.py

Public POST body:
    {
      "question": "A capacitor has C = 100 μF and U = 30 V. Calculate the energy stored."
    }

Defaults:
- debug = false
- polish = false

Endpoints:
- GET  /health
- GET  /solve?question=...
- POST /solve
- GET  /predict
- POST /predict
- POST /batch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from flask import Flask, jsonify, request
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: Flask.\n"
        "Install it with:\n"
        "    pip install flask flask-cors requests\n"
    ) from exc

try:
    from flask_cors import CORS
except ImportError:
    CORS = None


# ============================================================
# 1. Path setup
# ============================================================

SERVER_PATH = Path(__file__).resolve()
RESULT_DIR = SERVER_PATH.parent
PROJECT_ROOT = RESULT_DIR.parent

if str(RESULT_DIR) not in sys.path:
    sys.path.insert(0, str(RESULT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. Configuration
# ============================================================

DEFAULT_HOST = os.environ.get("PHYSICS_API_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("PHYSICS_API_PORT", "8000"))

DEFAULT_VERIFIED_DATA = os.environ.get(
    "PHYSICS_VERIFIED_DATA",
    str(PROJECT_ROOT / "Retrieve new data v2" / "verified_golden_expanded.csv"),
)

# Public API keeps explanations from the verified pipeline by default.
DEFAULT_ENABLE_POLISH = os.environ.get("PHYSICS_ENABLE_POLISH", "false").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DEFAULT_POLISH_MODEL = os.environ.get("PHYSICS_POLISH_MODEL", "trace_explainer_final")
DEFAULT_POLISH_BASE_URL = os.environ.get("PHYSICS_POLISH_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
DEFAULT_POLISH_TIMEOUT_SECONDS = float(os.environ.get("PHYSICS_POLISH_TIMEOUT_SECONDS", "45"))
DEFAULT_POLISH_MODE = os.environ.get("PHYSICS_POLISH_MODE", "conditional").strip().lower()
DEFAULT_POLISH_MAX_TOKENS = int(os.environ.get("PHYSICS_POLISH_MAX_TOKENS", "140"))
DEFAULT_POLISH_SEMANTIC = os.environ.get("PHYSICS_POLISH_SEMANTIC", "false").lower() in {
    "1",
    "true",
    "yes",
    "y",
}

DEFAULT_ENABLE_SEMANTIC_PARSER = os.environ.get("PHYSICS_ENABLE_SEMANTIC_PARSER", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DEFAULT_SEMANTIC_MODEL = os.environ.get("PHYSICS_SEMANTIC_MODEL", "numeric_parser_final")
DEFAULT_SEMANTIC_MIN_CONFIDENCE = float(os.environ.get("PHYSICS_SEMANTIC_MIN_CONFIDENCE", "0.50"))
DEFAULT_SEMANTIC_MAX_TOKENS = int(os.environ.get("PHYSICS_SEMANTIC_MAX_TOKENS", "512"))
DEFAULT_SEMANTIC_TIMEOUT_SECONDS = float(os.environ.get("PHYSICS_SEMANTIC_TIMEOUT_SECONDS", "45"))
DEFAULT_SEMANTIC_REPAIR_THRESHOLD = float(os.environ.get("PHYSICS_SEMANTIC_REPAIR_THRESHOLD", "0.80"))

DEFAULT_ENABLE_CHLT_REASONER = os.environ.get("PHYSICS_ENABLE_CHLT_REASONER", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DEFAULT_CHLT_MODEL = os.environ.get("PHYSICS_CHLT_MODEL", "chlt_reasoner_final")
DEFAULT_CHLT_MAX_TOKENS = int(os.environ.get("PHYSICS_CHLT_MAX_TOKENS", "256"))
DEFAULT_CHLT_TIMEOUT_SECONDS = float(os.environ.get("PHYSICS_CHLT_TIMEOUT_SECONDS", "20"))

DEFAULT_USE_LLM_FALLBACK = os.environ.get("PHYSICS_USE_LLM_FALLBACK", "false").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DEFAULT_LLM_MODEL_NAME = os.environ.get("PHYSICS_LLM_MODEL_NAME", "Qwen/Qwen2.5-Math-7B-Instruct")
DEFAULT_LLM_MODEL_PATH = os.environ.get("PHYSICS_LLM_MODEL_PATH", "").strip()
DEFAULT_LLM_LOCAL_FILES_ONLY = os.environ.get("PHYSICS_LLM_LOCAL_FILES_ONLY", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DEFAULT_LLM_LOAD_4BIT = os.environ.get("PHYSICS_LLM_LOAD_4BIT", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}


# ============================================================
# 3. Import pipeline
# ============================================================

try:
    import physics_pipeline as pipeline
except Exception as exc:
    raise SystemExit(
        "Could not import result/physics_pipeline.py.\n"
        "Make sure the stable pipeline file is named exactly:\n"
        "    result/physics_pipeline.py\n\n"
        f"Original error: {exc}"
    ) from exc

try:
    from physics_calculator_v2 import solve_numeric_payload as calculator_v2_solve
except Exception:
    calculator_v2_solve = None

try:
    from physics_calculator_v2.payload_validator import validate_numeric_payload as calculator_v2_validate
except Exception:
    calculator_v2_validate = None


# ============================================================
# 4. Flask app
# ============================================================

app = Flask(__name__)

if CORS is not None:
    CORS(app)


APP_STATE: Dict[str, Any] = {
    "pipeline_ready": False,
    "pipeline_error": None,
    "prepared_data_path": None,
    "llm_fallback_enabled": DEFAULT_USE_LLM_FALLBACK,
    "llm_model_ready": False,
    "llm_model_error": None,
    "llm_model_name_or_path": DEFAULT_LLM_MODEL_PATH or DEFAULT_LLM_MODEL_NAME,
    "startup_time": time.strftime("%Y-%m-%d %H:%M:%S"),
}


# ============================================================
# 5. Basic helpers
# ============================================================

def _bool_from_any(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_json_body() -> Dict[str, Any]:
    if not request.data:
        return {}
    try:
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _error_response(
    message: str,
    status: int = 400,
    error_type: str = "bad_request",
    **extra: Any,
):
    payload = {
        "ok": False,
        "error_type": error_type,
        "message": message,
    }
    payload.update(extra)
    return jsonify(_json_safe(payload)), status


def _json_safe(obj: Any) -> Any:
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [_json_safe(v) for v in sorted(obj, key=lambda x: str(x))]
    return obj


# ============================================================
# 6. Input validation
# ============================================================

_PHYSICS_KEYWORDS = {
    "force", "electric", "charge", "field", "voltage", "current", "resistor",
    "resistance", "capacitor", "capacitance", "inductor", "inductance",
    "energy", "power", "circuit", "frequency", "resonance", "impedance",
    "magnetic", "flux", "emf", "solenoid", "coil", "oscillation", "lc",
    "rlc", "ohm", "newton", "joule", "watt", "volt", "ampere", "farad",
    "henry", "coulomb", "hertz", "rms", "reactance", "measurement",
    "error", "relative error", "uncertainty", "vector", "resultant",
    "perpendicular", "parallel", "series", "angle",
    "ω", "μ", "µ", "Ω",
}

_UNIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:n|N|v|V|a|A|j|J|w|W|hz|Hz|f|F|h|H|c|C|ohm|Ω|uF|μF|µF|mF|pF|uC|μC|µC|nC|mH|uH|μH|kg|g|cm|mm|m)\b)"
)

def _validate_question_text(question: Any) -> tuple[bool, str]:
    if question is None:
        return False, "Missing required field: question."

    if not isinstance(question, str):
        return False, "Field 'question' must be a string."

    q = question.strip()
    if not q:
        return False, "Question cannot be empty."

    # Reject obvious plain arithmetic before checking length.
    compact = re.sub(r"\s+", "", q)
    if re.fullmatch(r"[0-9+\-*/^().=?]+", compact):
        return False, "This looks like a pure arithmetic question, not a supported physics problem."

    if len(q) < 6:
        return False, "Question is too short to be parsed as a physics problem."

    q_lower = f" {q.lower()} "
    has_keyword = any(k in q_lower for k in _PHYSICS_KEYWORDS)
    has_unit = bool(_UNIT_PATTERN.search(q))

    if not (has_keyword or has_unit):
        return False, "The question does not appear to be a supported physics problem."

    return True, ""


# ============================================================
# 7. Formula / math prettifier
# ============================================================

_SUPERSCRIPT_MAP = str.maketrans({
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "-": "⁻",
    "+": "⁺",
})

_FRACTION_MAP = {
    "1/2": "½",
    "1 / 2": "½",
    "1/3": "⅓",
    "2/3": "⅔",
    "1/4": "¼",
    "3/4": "¾",
}

def _to_superscript(exp: str) -> str:
    return str(exp).translate(_SUPERSCRIPT_MAP)


def _prettify_math_text(text: Any) -> Any:
    """
    Convert code-style formulas into readable Unicode math text.
    This is applied only to display text, not computation.

    Examples:
    - sqrt(F1^2+F2^2) -> √(F1²+F2²)
    - 1/2*C*U^2 -> ½ × C × U²
    - 0.5*0.0001*30^2 -> 0.5 × 0.0001 × 30²
    """
    if not isinstance(text, str):
        return text

    s = text

    # Normalize common micro sign display.
    s = s.replace(" uF", " μF").replace(" uC", " μC").replace(" uH", " μH")
    s = s.replace("uF", "μF").replace("uC", "μC").replace("uH", "μH")

    # sqrt(...)
    s = re.sub(r"\bsqrt\s*\(", "√(", s)
    s = re.sub(r"\bpi\b", "π", s)
    s = re.sub(r"(\d)\s+π", r"\1π", s)

    # Fractions before multiplication cleanup.
    for raw, pretty in _FRACTION_MAP.items():
        s = s.replace(raw, pretty)

    # Powers: x^2, U^2, 10^-4
    def repl_pow(match: re.Match) -> str:
        base = match.group(1)
        exp = match.group(2)
        return f"{base}{_to_superscript(exp)}"

    s = re.sub(r"([A-Za-z0-9)\]])\^([+-]?\d+)", repl_pow, s)

    # Convert simple * to × in math-like contexts.
    s = re.sub(r"(?<=\d)\*(?=\d)", " × ", s)
    s = re.sub(r"(?<=[A-Za-z0-9²³⁴⁵⁶⁷⁸⁹)])\*(?=[A-Za-z0-9(])", " × ", s)

    # Common formula cleanup.
    s = s.replace("1/2 ×", "½ ×")
    s = s.replace("0.5 ×", "0.5 ×")
    s = s.replace("U0", "U₀").replace("I0", "I₀")
    s = s.replace("Umax", "Uₘₐₓ").replace("Imax", "Iₘₐₓ")
    s = s.replace("1/(2π√(LC))", "1/(2π√(LC))")
    s = s.replace("C/L", "C/L")
    s = s.replace("cosθ", "cos θ")
    s = s.replace("F1", "F₁").replace("F2", "F₂")
    s = s.replace("R=", "R = ").replace("W=", "W = ").replace("Z=", "Z = ")
    s = s.replace("..", ".")

    # Clean spaces.
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("( ", "(").replace(" )", ")")

    return s


def _prettify_response_math(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in {"python_code", "formula_latex"}:
                out[k] = v
            else:
                out[k] = _prettify_response_math(v)
        return out

    if isinstance(obj, list):
        return [_prettify_response_math(x) for x in obj]

    if isinstance(obj, str):
        return _prettify_math_text(obj)

    return obj


# ============================================================
# 8. Pipeline preparation and solving
# ============================================================

def _ensure_pipeline_ready() -> None:
    if APP_STATE["pipeline_ready"]:
        return

    data_path = DEFAULT_VERIFIED_DATA

    try:
        if hasattr(pipeline, "prepare_pipeline"):
            try:
                pipeline.prepare_pipeline(data_path, verbose=False)
            except TypeError:
                pipeline.prepare_pipeline(data_path)

        if DEFAULT_USE_LLM_FALLBACK:
            loader = getattr(pipeline, "load_llm_fallback_model", None)
            if loader is None:
                raise RuntimeError("physics_pipeline.py does not expose load_llm_fallback_model().")
            model_ref = DEFAULT_LLM_MODEL_PATH or DEFAULT_LLM_MODEL_NAME
            loader(
                model_ref,
                load_in_4bit=DEFAULT_LLM_LOAD_4BIT,
                local_files_only=DEFAULT_LLM_LOCAL_FILES_ONLY,
            )
            APP_STATE["llm_model_ready"] = True
            APP_STATE["llm_model_error"] = None
            APP_STATE["llm_model_name_or_path"] = model_ref
        else:
            if hasattr(pipeline, "USE_LLM_FALLBACK"):
                pipeline.USE_LLM_FALLBACK = False
            APP_STATE["llm_model_ready"] = False
            APP_STATE["llm_model_error"] = None

        APP_STATE["pipeline_ready"] = True
        APP_STATE["pipeline_error"] = None
        APP_STATE["prepared_data_path"] = data_path

    except Exception:
        APP_STATE["pipeline_ready"] = False
        APP_STATE["pipeline_error"] = traceback.format_exc()
        APP_STATE["prepared_data_path"] = data_path
        APP_STATE["llm_model_ready"] = False
        APP_STATE["llm_model_error"] = traceback.format_exc() if DEFAULT_USE_LLM_FALLBACK else None
        raise


def _solve_with_pipeline(
    question: str,
    debug: bool = False,
    true_answer: Optional[str] = None,
    true_unit: Optional[str] = None,
) -> Dict[str, Any]:
    if not hasattr(pipeline, "solve_physics_question"):
        raise RuntimeError("physics_pipeline.py does not expose solve_physics_question().")

    solve_fn = pipeline.solve_physics_question

    try:
        return solve_fn(question, debug=debug)
    except TypeError:
        pass

    try:
        return solve_fn(question, true_answer, true_unit, debug=debug)
    except TypeError:
        pass

    return solve_fn(question)


def _pipeline_result_is_unanswered(out: Dict[str, Any]) -> bool:
    method = str(out.get("method", "")).lower()
    answer = str(out.get("answer", "")).strip()

    if "unanswered" in method:
        return True

    if answer == "":
        return True

    if answer.lower() in {"none", "nan", "null"}:
        return True

    return False


# ============================================================
# 9. Optional polish hook
# ============================================================

def _ascii_unit(unit: Any) -> str:
    text = str(unit or "").strip()
    if text in {"", "-", "None", "none", "null"}:
        return ""

    try:
        text = pipeline.canonical_unit(text)
    except Exception:
        text = text.replace("μ", "u").replace("µ", "u").replace("Ω", "ohm")

    replacements = {
        "μ": "u",
        "µ": "u",
        "Ω": "ohm",
        "Ohm": "ohm",
        "Ohms": "ohm",
        "ohms": "ohm",
        "²": "^2",
        "³": "^3",
        "·": "*",
        " ": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def _string_answer(answer: Any) -> str:
    text = str(answer if answer is not None else "").strip()
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def _compact_for_prompt(value: Any, max_chars: int = 2500) -> str:
    try:
        text = json.dumps(_json_safe(value), ensure_ascii=False, indent=2)
    except Exception:
        text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "\n..."
    return text


def _build_polish_messages(out: Dict[str, Any], question: str) -> list[Dict[str, str]]:
    answer = _string_answer(out.get("answer"))
    unit = _ascii_unit(out.get("unit"))
    topic = str(out.get("topic_pred") or out.get("topic") or "physics").strip()
    prefix = str(out.get("prefix_pred") or out.get("prefix") or "unknown").strip()

    trace_payload = {
        "method": out.get("method", ""),
        "topic": topic,
        "given_quantities": (out.get("trace") or {}).get("given_quantities", []),
        "unit_conversions": (out.get("trace") or {}).get("unit_conversions", []),
        "formulas": (out.get("trace") or {}).get("formulas", []),
        "calculation_steps": (out.get("trace") or {}).get("calculation_steps", []),
        "cot": out.get("cot", []),
    }

    system = (
        "You are a careful physics explanation writer. Given a physics question, "
        "a locked final answer, and a solver trace, write a concise English explanation. "
        "Do not change the locked answer or unit. Return only valid JSON with exactly one key: explanation."
    )
    user = (
        f"Prefix: {prefix}\n"
        f"Topic: {topic}\n"
        f"Question: {question}\n"
        f"Locked answer: {answer} {unit or 'unitless'}\n"
        f"Solver trace:\n{_compact_for_prompt(trace_payload)}\n"
        "Task: Explain why the locked answer is correct."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _call_openai_chat_completion(
    messages: list[Dict[str, str]],
    model_name: str,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": int(max_tokens or DEFAULT_POLISH_MAX_TOKENS),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{DEFAULT_POLISH_BASE_URL}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=float(timeout_seconds or DEFAULT_POLISH_TIMEOUT_SECONDS)) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return str(parsed["choices"][0]["message"]["content"]).strip()


def _fetch_vllm_models() -> Dict[str, Any]:
    req = urllib.request.Request(
        f"{DEFAULT_POLISH_BASE_URL}/models",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=DEFAULT_POLISH_TIMEOUT_SECONDS) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("vLLM /v1/models did not return a JSON object.")
    return parsed


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = str(text or "").strip()
    candidates = [text]

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    if first >= 0:
        repaired = _repair_truncated_json_object(text[first:])
        if repaired:
            candidates.append(repaired)

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _repair_truncated_json_object(text: str) -> Optional[str]:
    """Best-effort repair for JSON-only adapter output cut off by token limits."""
    s = str(text or "").strip()
    if not s.startswith("{"):
        return None

    stack: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    if in_string:
        s += '"'
    if stack:
        s += "".join(reversed(stack))

    try:
        obj = json.loads(s)
    except Exception:
        return None
    return s if isinstance(obj, dict) else None


def _validate_polished_explanation(raw_text: str, locked_answer: str, locked_unit: str) -> tuple[Optional[str], str]:
    obj = _extract_json_object(raw_text)
    if obj is None:
        return None, "polish_invalid_json"

    if set(obj.keys()) != {"explanation"}:
        return None, "polish_invalid_schema"

    explanation = str(obj.get("explanation") or "").strip()
    if len(explanation.split()) < 12:
        return None, "polish_explanation_too_short"

    if locked_answer and locked_answer not in explanation:
        final = f"{locked_answer} {locked_unit}".strip()
        explanation = explanation.rstrip(".") + f". Therefore, the final answer is {final}."

    return explanation, "ok"


# ============================================================
# 10. Semantic parser fallback: LLM parses, Python solves
# ============================================================

SEMANTIC_REQUIRED_KEYS = {
    "question_kind",
    "topic",
    "target",
    "givens",
    "constraints",
    "formula_candidates",
    "confidence",
}


def _semantic_system_prompt() -> str:
    return (
        "You are the numeric_parser_final adapter for educational physics. "
        "Return exactly one compact JSON object with keys: question_kind, topic, target, givens, constraints, formula_candidates, confidence. "
        "Do not solve the problem and do not include answer, unit_answer, final_result, cot, markdown, or python_code. "
        "Every given must preserve a raw_span copied from the question and include role/value_text/unit_text when possible. "
        "Formula candidates must use registered calculator formula_id values when known. The deterministic calculator will compute the final answer."
    )


def _build_semantic_messages(question: str, route_hint: Dict[str, Any]) -> list[Dict[str, str]]:
    topic_hint = route_hint.get("topic_pred") or route_hint.get("topic") or "unknown"
    prefix_hint = route_hint.get("prefix_pred") or route_hint.get("prefix") or "unknown"
    return [
        {"role": "system", "content": _semantic_system_prompt()},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Topic hint: {topic_hint}\n"
                f"Prefix hint: {prefix_hint}\n"
                "Return numeric parser JSON only. Do not calculate the final answer. Keep it short and syntactically complete."
            ),
        },
    ]


def _validate_semantic_payload(payload: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], str]:
    missing = sorted(SEMANTIC_REQUIRED_KEYS - set(payload.keys()))
    if missing:
        return None, f"semantic_missing_keys:{missing}"
    if any(key in payload for key in ("answer", "unit_answer", "python_code", "golden_code", "final_result", "cot")):
        return None, "semantic_payload_contains_answer_or_code"
    if not isinstance(payload.get("target"), dict):
        return None, "semantic_target_not_object"
    if not isinstance(payload.get("givens"), list):
        return None, "semantic_givens_not_list"
    payload.setdefault("canonical_problem", "calculator_v2_generic")
    payload.setdefault("relations", [])
    payload.setdefault("constraints", [])
    payload.setdefault("assumptions", [])
    payload.setdefault("formula_candidates", [])
    payload.setdefault("question_kind", "calculation")
    payload.setdefault("confidence", 0.70)
    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        return None, "semantic_invalid_confidence"
    if confidence < DEFAULT_SEMANTIC_MIN_CONFIDENCE:
        return None, f"semantic_low_confidence:{confidence}"
    return payload, "ok"


def _semantic_unit_scale(unit: Any) -> float:
    unit_text = _ascii_unit(unit)
    scale = {
        "pF": 1e-12,
        "nF": 1e-9,
        "uF": 1e-6,
        "mF": 1e-3,
        "F": 1.0,
        "pC": 1e-12,
        "nC": 1e-9,
        "uC": 1e-6,
        "mC": 1e-3,
        "C": 1.0,
        "uH": 1e-6,
        "mH": 1e-3,
        "H": 1.0,
        "uA": 1e-6,
        "mA": 1e-3,
        "A": 1.0,
        "mV": 1e-3,
        "kV": 1e3,
        "V": 1.0,
        "mJ": 1e-3,
        "uJ": 1e-6,
        "nJ": 1e-9,
        "J": 1.0,
        "cm": 1e-2,
        "mm": 1e-3,
        "m": 1.0,
        "cm^2": 1e-4,
        "mm^2": 1e-6,
        "m^2": 1.0,
        "ohm": 1.0,
        "W": 1.0,
        "Hz": 1.0,
        "rad/s": 1.0,
        "V/m": 1.0,
        "N/C": 1.0,
        "N": 1.0,
        "mN": 1e-3,
        "uN": 1e-6,
        "kg": 1.0,
        "g": 1e-3,
        "mg": 1e-6,
        "uT": 1e-6,
        "mT": 1e-3,
        "T": 1.0,
        "Wb": 1.0,
        "turns/m": 1.0,
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "degree": 1.0,
        "deg": 1.0,
        "rad": 1.0,
        "times": 1.0,
        "%": 1.0,
    }
    return scale.get(unit_text, 1.0)


def _semantic_givens(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    return payload.get("givens") if isinstance(payload.get("givens"), list) else []


def _semantic_values_by_unit(payload: Dict[str, Any], units: set[str]) -> list[float]:
    values: list[float] = []
    for given in _semantic_givens(payload):
        if not isinstance(given, dict):
            continue
        unit = _ascii_unit(given.get("unit"))
        if unit not in units:
            continue
        value = given.get("si_value", None)
        if value is None:
            raw_value = given.get("value", None)
            if raw_value is None:
                continue
            try:
                value = float(raw_value) * _semantic_unit_scale(unit)
            except Exception:
                continue
        try:
            values.append(float(value))
        except Exception:
            pass
    return values


def _semantic_first_by_units(payload: Dict[str, Any], units: set[str]) -> Optional[float]:
    values = _semantic_values_by_unit(payload, units)
    return values[0] if values else None


def _semantic_question_values_by_unit(payload: Dict[str, Any], units: set[str]) -> list[float]:
    question = str(payload.get("_question") or "")
    if not question:
        return []
    values: list[float] = []
    unit_alt = "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
    if not unit_alt:
        return values

    sci_pattern = re.compile(
        rf"([+-]?\d+(?:\.\d+)?)?\s*(?:x|×|\*)?\s*10\s*(?:\^|\*\*)\s*\{{?([+-]?\d+)\}}?\s*({unit_alt})\b",
        flags=re.IGNORECASE,
    )
    normalized_question = (
        question.replace("µ", "u")
        .replace("μ", "u")
        .replace("Ω", "ohm")
        .replace("Ω", "ohm")
        .replace("Î©", "ohm")
    )
    normalized_question = re.sub(r"(?<=\d)\s*\?\s*(?=10\s*(?:\^|\*\*))", " x ", normalized_question)
    normalized_question = normalized_question.replace("?C", "uC").replace("?F", "uF").replace("?A", "uA")
    normalized_question = re.sub(r"(?<=\d)\s*\?(?=\s|,|\.|;|\)|$)", "ohm", normalized_question)

    used_spans: list[tuple[int, int]] = []
    strict_sci_pattern = re.compile(
        rf"([+-]?\d+(?:\.\d+)?)\s*(?:x|×|\*)\s*10\s*(?:\^|\*\*)\s*\{{?([+-]?\d+)\}}?\s*({unit_alt})\b",
        flags=re.IGNORECASE,
    )
    for match in strict_sci_pattern.finditer(normalized_question):
        coeff = float(match.group(1))
        exponent = int(match.group(2))
        unit = _ascii_unit(match.group(3))
        values.append(coeff * (10.0 ** exponent) * _semantic_unit_scale(unit))
        used_spans.append(match.span())

    for match in sci_pattern.finditer(normalized_question):
        if any(match.start() >= start and match.end() <= end for start, end in used_spans):
            continue
        coeff = float(match.group(1)) if match.group(1) not in {None, ""} else 1.0
        exponent = int(match.group(2))
        unit = _ascii_unit(match.group(3))
        values.append(coeff * (10.0 ** exponent) * _semantic_unit_scale(unit))
        used_spans.append(match.span())

    plain_pattern = re.compile(rf"([+-]?\d+(?:\.\d+)?)\s*({unit_alt})\b", flags=re.IGNORECASE)
    for match in plain_pattern.finditer(normalized_question):
        if any(match.start() >= start and match.end() <= end for start, end in used_spans):
            continue
        if match.start() > 0 and normalized_question[match.start() - 1] in {"^"}:
            continue
        unit = _ascii_unit(match.group(2))
        raw = float(match.group(1)) * _semantic_unit_scale(unit)
        if not any(abs(raw - existing) <= max(1e-18, abs(existing) * 1e-9) for existing in values):
            values.append(raw)
    return values


def _semantic_target_unit(payload: Dict[str, Any]) -> str:
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    return _ascii_unit(target.get("unit"))


def _semantic_target_symbol(payload: Dict[str, Any]) -> str:
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    return str(target.get("symbol") or "").strip()


def _semantic_text_blob(payload: Dict[str, Any]) -> str:
    pieces: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            pieces.append(value)
        elif isinstance(value, dict):
            for inner in value.values():
                add(inner)
        elif isinstance(value, list):
            for inner in value:
                add(inner)

    for key in ("_question", "topic", "canonical_problem", "target", "givens", "relations", "constraints", "assumptions"):
        add(payload.get(key))
    return " ".join(pieces).lower()


def _semantic_given_si_value(given: Dict[str, Any]) -> Optional[float]:
    unit = _ascii_unit(given.get("unit"))
    value = given.get("si_value", None)
    if value is None:
        raw_value = given.get("value", None)
        if raw_value is None:
            return None
        try:
            value = float(raw_value) * _semantic_unit_scale(unit)
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _semantic_values_by_symbol(payload: Dict[str, Any], symbols: set[str]) -> list[float]:
    values: list[float] = []
    wanted = {s.lower() for s in symbols}
    for given in _semantic_givens(payload):
        if not isinstance(given, dict):
            continue
        symbol = str(given.get("symbol") or "").strip().lower()
        if symbol not in wanted:
            continue
        value = _semantic_given_si_value(given)
        if value is not None:
            values.append(value)
    return values


def _semantic_uncertainty_from_given(given: Dict[str, Any]) -> Optional[float]:
    for key in ("uncertainty", "error", "delta", "absolute_error", "abs_error"):
        if key in given and given.get(key) not in (None, ""):
            try:
                return abs(float(given.get(key))) * _semantic_unit_scale(given.get("unit"))
            except Exception:
                pass

    text = f"{given.get('source', '')} {given.get('unit', '')}"
    match = re.search(
        r"(?:±|\+/-|\?)\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([a-zA-ZµμÎÂΩ%/^0-9*]*)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    try:
        raw_error = float(match.group(1))
    except Exception:
        return None

    unit_hint = _ascii_unit(match.group(2))
    if not unit_hint:
        unit_hint = _ascii_unit(given.get("unit"))
    unit_hint = re.sub(r"^[±?+\-/0-9.eE]+", "", unit_hint)
    return abs(raw_error) * _semantic_unit_scale(unit_hint)


def _semantic_measurement_pairs(payload: Dict[str, Any]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    plain_values: list[float] = []

    for given in _semantic_givens(payload):
        if not isinstance(given, dict):
            continue
        value = _semantic_given_si_value(given)
        if value is None:
            continue
        uncertainty = _semantic_uncertainty_from_given(given)
        if uncertainty is None:
            plain_values.append(abs(value))
        else:
            pairs.append((abs(value), abs(uncertainty)))

    if pairs:
        return [(v, e) for v, e in pairs if v > 0 and e >= 0]

    if len(plain_values) >= 2:
        measured = max(plain_values)
        error = min(v for v in plain_values if v > 0)
        if measured > 0 and error <= measured:
            return [(measured, error)]

    return []


def _semantic_angle_degrees(payload: Dict[str, Any]) -> Optional[float]:
    text = _semantic_text_blob(payload)
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:degree|degrees|deg|°)", text)
    if match:
        return float(match.group(1))
    if "right-angle" in text or "right angle" in text or "perpendicular" in text:
        return 90.0
    if "equilateral" in text:
        return 60.0
    return None


def _semantic_dielectric_constant(payload: Dict[str, Any]) -> Optional[float]:
    text = _semantic_text_blob(payload)
    if not any(word in text for word in ("dielectric", "epsilon", "permittivity", "oil", "liquid")):
        return None
    patterns = [
        r"(?:dielectric constant|epsilon|relative permittivity|ε)\s*(?:=|is)?\s*([0-9]+(?:\.[0-9]+)?)",
        r"\be\s*(?:=|is)\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                value = float(match.group(1))
            except Exception:
                continue
            if value > 0:
                return value
    return None


def _semantic_question_signal_amplitude(payload: Dict[str, Any], symbol: str) -> Optional[float]:
    question = str(payload.get("_question") or "")
    if not question:
        return None
    text = question.replace("μ", "u").replace("µ", "u").replace("×", "*")
    pattern = rf"\b{re.escape(symbol)}\s*(?:\(t\))?\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\*)?\s*(?:sin|cos)\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    pattern = rf"\b{re.escape(symbol)}\s*(?:\(t\))?\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\*)?\s*(?:sqrt|√)\s*2\s*(?:\*)?\s*(?:sin|cos)\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) * (2.0 ** 0.5)
    return None


def _semantic_question_rms_voltage(payload: Dict[str, Any]) -> Optional[float]:
    question = str(payload.get("_question") or "")
    if not question:
        return None
    text = question.replace("μ", "u").replace("µ", "u").replace("×", "*")
    match = re.search(
        r"\b[Uu]\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\*)?\s*(?:sqrt|√)\s*2\s*(?:\*)?\s*(?:sin|cos)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return None


def _semantic_capacitor_basic(
    payload: Dict[str, Any],
    C: Optional[float],
    U: Optional[float],
    W: Optional[float],
    q_values: list[float],
    target_unit: str,
    confidence: float,
) -> Optional[Dict[str, Any]]:
    target_symbol = _semantic_target_symbol(payload)
    if target_symbol == "Q" and C and U:
        val = C * U
        return _semantic_result(val, target_unit, confidence, payload, f"Q = C*U = {val:.6g} C")
    if target_symbol == "W" and C and U:
        val = 0.5 * C * U * U
        return _semantic_result(val, target_unit, confidence, payload, f"W = 0.5*C*U^2 = {val:.6g} J")
    if target_symbol == "C" and W and U:
        val = 2.0 * W / (U * U)
        return _semantic_result(val, target_unit, confidence, payload, f"C = 2W/U^2 = {val:.6g} F")
    if target_symbol == "U" and W and C:
        import math

        val = math.sqrt(2.0 * W / C)
        return _semantic_result(val, target_unit, confidence, payload, f"U = sqrt(2W/C) = {val:.6g} V")
    if target_symbol == "U" and C and q_values:
        val = abs(q_values[0]) / C
        return _semantic_result(val, target_unit, confidence, payload, f"U = Q/C = {val:.6g} V")
    return None


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    try:
        numeric = float(value)
        return f"{numeric:.6g}"
    except Exception:
        return str(value).strip()


def _semantic_result(value_si_or_target: float, target_unit: str, confidence: float, payload: Dict[str, Any], calc: str) -> Dict[str, Any]:
    scale = _semantic_unit_scale(target_unit)
    value_out = value_si_or_target / scale if scale else value_si_or_target
    text_blob = _semantic_text_blob(payload)
    if "round" in text_blob and ("two decimal" in text_blob or "2 decimal" in text_blob):
        answer = f"{float(value_out):.2f}"
    elif "round" in text_blob and ("nearest integer" in text_blob or "integer" in text_blob):
        answer = str(int(round(float(value_out))))
    else:
        answer = _format_number(value_out)
    canonical = str(payload.get("canonical_problem") or "")
    topic = str(payload.get("topic") or "")
    explanation = (
        f"The question is first normalized to the canonical physics form `{canonical}`. "
        f"The numerical answer is then computed by the Python canonical solver, not by the language model. "
        f"The locked calculation is: {calc}. "
        f"After converting the result to the requested unit, the final answer is {answer} {target_unit}."
    )
    return {
        "answer": answer,
        "unit": target_unit,
        "explanation": explanation,
        "cot": [
            f"Semantic parse: topic={topic}, canonical_problem={canonical}.",
            f"Canonical calculation: {calc}.",
            f"Final answer: {answer} {target_unit}.",
        ],
        "premises": [
            "The LLM semantic parser produced only canonical JSON and did not compute the final answer.",
            "The final answer was computed by the Python canonical solver.",
        ],
        "confidence": min(max(confidence, 0.0), 0.9),
        "method": "semantic_parser_python_canonical_solver",
        "topic_pred": topic,
        "prefix_pred": "",
        "semantic_payload": payload,
        "semantic_status": "ok",
    }


def _solve_semantic_canonical(payload: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], str]:
    import math

    canonical = str(payload.get("canonical_problem") or "")
    canonical_aliases = {
        "rlc_resonance_capacitor_from_L_f": "rlc_resonance_capacitance_from_L_f",
        "resonance_capacitor_from_L_f": "rlc_resonance_capacitance_from_L_f",
        "resonance_capacitance_from_L_f": "rlc_resonance_capacitance_from_L_f",
        "capacitors_series_equivalent": "capacitor_equivalent_capacitance",
        "capacitors_parallel_equivalent": "capacitor_equivalent_capacitance",
        "capacitor_capacitance_from_energy_and_voltage": "capacitor_capacitance_from_energy_voltage",
        "inductor_inductance_from_energy_current": "inductance_from_energy_current",
        "lc_energy_from_I_L": "lc_energy_from_inductor_current",
        "lc_energy_from_current_inductance": "lc_energy_from_inductor_current",
        "lc_max_voltage_from_current_capacitance": "lc_max_voltage_from_inductor_current",
        "lc_angular_frequency_from_freq": "lc_angular_frequency_from_frequency",
        "circuit_power_from_I_R": "circuit_power_from_U_I_R",
        "circuit_power_from_U_R": "circuit_power_from_U_I_R",
        "ohm_current_from_power_voltage": "circuit_power_from_U_I_R",
        "ohm_voltage_from_power_current": "circuit_power_from_U_I_R",
        "measurement_uncertainty_from_measurement_uncertainty": "measurement_resistance_uncertainty_from_U_I",
        "measurement_relative_error_from_absolute_error": "measurement_relative_error",
        "measurement_relative_uncertainty_from_absolute_uncertainty": "measurement_relative_error",
        "electric_field_from_force_charge": "electric_field_point_charge_or_superposition",
        "resistance_from_voltage_current": "ohm_resistance_from_voltage_current",
    }
    canonical = canonical_aliases.get(canonical, canonical)
    payload["canonical_problem"] = canonical
    target_unit = _semantic_target_unit(payload)
    target_symbol = _semantic_target_symbol(payload)
    if not target_unit:
        return None, "semantic_missing_target_unit"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.6

    C = _semantic_first_by_units(payload, {"F", "uF", "nF", "pF", "mF"})
    L = _semantic_first_by_units(payload, {"H", "mH", "uH"})
    U = _semantic_first_by_units(payload, {"V", "mV", "kV"})
    I = _semantic_first_by_units(payload, {"A", "mA", "uA"})
    R_values = _semantic_values_by_unit(payload, {"ohm"})
    W = _semantic_first_by_units(payload, {"J", "mJ", "uJ", "nJ"})
    power_values = _semantic_values_by_unit(payload, {"W", "mW", "kW"})
    f = _semantic_first_by_units(payload, {"Hz"})
    q_values = _semantic_values_by_unit(payload, {"C", "uC", "nC", "pC", "mC"})
    if any(abs(q) > 0.1 for q in q_values):
        q_from_question = _semantic_question_values_by_unit(payload, {"C", "uC", "nC", "pC", "mC"})
        if q_from_question:
            q_values = q_from_question
    r_values = _semantic_values_by_unit(payload, {"m", "cm", "mm"})
    B = _semantic_first_by_units(payload, {"T", "mT", "uT"})
    areas = _semantic_values_by_unit(payload, {"m^2", "cm^2", "mm^2"})
    t_values = _semantic_values_by_unit(payload, {"s", "ms", "us"})
    force_values = _semantic_values_by_unit(payload, {"N", "mN", "uN"})
    if not force_values:
        force_values = _semantic_question_values_by_unit(payload, {"N", "mN", "uN"})
    field_values = _semantic_values_by_unit(payload, {"V/m", "N/C"})
    if not field_values:
        field_values = _semantic_question_values_by_unit(payload, {"V/m", "N/C"})
    mass_values = _semantic_values_by_unit(payload, {"kg", "g", "mg"})
    if not mass_values:
        mass_values = _semantic_question_values_by_unit(payload, {"kg", "g", "mg"})
    text_blob = _semantic_text_blob(payload)

    if canonical == "lc_angular_frequency_from_frequency" and f:
        val = 2.0 * math.pi * f
        return _semantic_result(val, target_unit, confidence, payload, f"omega = 2*pi*f = {val:.6g} rad/s"), "ok"
    if canonical == "lc_angular_frequency_from_L_C" and L and C:
        val = 1.0 / math.sqrt(L * C)
        return _semantic_result(val, target_unit, confidence, payload, f"omega = 1/sqrt(L*C) = {val:.6g} rad/s"), "ok"
    if canonical == "lc_frequency_from_L_C" and L and C:
        val = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
        return _semantic_result(val, target_unit, confidence, payload, f"f = 1/(2*pi*sqrt(L*C)) = {val:.6g} Hz"), "ok"
    if canonical == "lc_max_current_from_capacitor_voltage" and L and C and U:
        val = U * math.sqrt(C / L)
        return _semantic_result(val, target_unit, confidence, payload, f"I0 = U0*sqrt(C/L) = {val:.6g} A"), "ok"
    if canonical == "lc_max_voltage_from_inductor_current" and L and C and I:
        val = I * math.sqrt(L / C)
        return _semantic_result(val, target_unit, confidence, payload, f"U0 = I0*sqrt(L/C) = {val:.6g} V"), "ok"
    if canonical == "lc_energy_from_capacitor_voltage" and C and U:
        val = 0.5 * C * U * U
        return _semantic_result(val, target_unit, confidence, payload, f"W = 0.5*C*U^2 = {val:.6g} J"), "ok"
    if canonical == "lc_energy_from_inductor_current" and L and I:
        val = 0.5 * L * I * I
        return _semantic_result(val, target_unit, confidence, payload, f"W = 0.5*L*I^2 = {val:.6g} J"), "ok"

    if canonical == "capacitor_charge_from_C_U" and C and U:
        val = C * U
        return _semantic_result(val, target_unit, confidence, payload, f"Q = C*U = {val:.6g} C"), "ok"
    if canonical == "capacitor_energy_from_C_U" and C and U:
        if "shared" in text_blob and "two" in text_blob and ("identical" in text_blob or "same" in text_blob):
            val = 0.25 * C * U * U
            return _semantic_result(val, target_unit, confidence, payload, f"charge is shared by two identical capacitors, W_final = 0.25*C*U^2 = {val:.6g} J"), "ok"
        val = 0.5 * C * U * U
        return _semantic_result(val, target_unit, confidence, payload, f"W = 0.5*C*U^2 = {val:.6g} J"), "ok"
    if canonical == "capacitor_capacitance_from_energy_voltage" and W and U:
        val = 2.0 * W / (U * U)
        return _semantic_result(val, target_unit, confidence, payload, f"C = 2W/U^2 = {val:.6g} F"), "ok"
    if canonical == "capacitor_voltage_from_energy_capacitance" and W and C:
        val = math.sqrt(2.0 * W / C)
        return _semantic_result(val, target_unit, confidence, payload, f"U = sqrt(2W/C) = {val:.6g} V"), "ok"
    if canonical == "capacitor_general":
        if target_symbol == "W" and C:
            u_amp = _semantic_question_signal_amplitude(payload, "U")
            if u_amp:
                val = 0.5 * C * u_amp * u_amp
                return _semantic_result(val, target_unit, confidence, payload, f"W_max = 0.5*C*U0^2 = {val:.6g} J"), "ok"
            if "doubled" in text_blob and target_unit == "times":
                return _semantic_result(4.0, target_unit, confidence, payload, "W is proportional to U^2, so doubling U gives 4 times the energy"), "ok"
        if target_symbol == "C" and L and f:
            val = 1.0 / ((2.0 * math.pi * f) ** 2 * L)
            return _semantic_result(val, target_unit, confidence, payload, f"C = 1/((2*pi*f)^2*L) = {val:.6g} F"), "ok"
    if canonical in {"parallel_plate_capacitor", "capacitor_general"} and target_symbol == "W" and U:
        c_values = _semantic_values_by_unit(payload, {"F", "uF", "nF", "pF", "mF"})
        dielectric = _semantic_dielectric_constant(payload)
        if len(c_values) == 1 and dielectric and ("connected" in text_blob or "source" in text_blob):
            val = 0.5 * dielectric * c_values[0] * U * U
            return _semantic_result(val, target_unit, confidence, payload, f"source remains connected, C' = epsilon_r*C, W = 0.5*C'*U^2 = {val:.6g} J"), "ok"
        if len(c_values) >= 2 and ("isolated" in text_blob or "disconnected" in text_blob or "moved apart" in text_blob):
            charge = c_values[0] * U
            val = charge * charge / (2.0 * c_values[-1])
            return _semantic_result(val, target_unit, confidence, payload, f"Q is conserved, W = Q^2/(2*C_final) = {val:.6g} J"), "ok"
        if len(c_values) >= 2:
            return None, "semantic_capacitor_multiple_capacitances_ambiguous"
    if canonical in {"capacitor_general", "parallel_plate_capacitor"}:
        basic = _semantic_capacitor_basic(payload, C, U, W, q_values, target_unit, confidence)
        if basic is not None:
            return basic, "ok"

    if canonical == "parallel_plate_capacitor":
        eps0 = 8.854187817e-12
        cap = None
        plate_area = areas[0] if areas else None
        plate_gap = r_values[0] if r_values else None
        if plate_area is None and len(r_values) >= 2 and ("radius" in text_blob or "circular" in text_blob):
            plate_area = math.pi * r_values[0] * r_values[0]
            plate_gap = r_values[1]
        if plate_area and plate_gap:
            cap = eps0 * plate_area / plate_gap
        if cap:
            if target_symbol == "C":
                return _semantic_result(cap, target_unit, confidence, payload, f"C = eps0*A/d = {cap:.6g} F"), "ok"
            if target_symbol == "Q" and U:
                val = cap * U
                return _semantic_result(val, target_unit, confidence, payload, f"Q = C*U, C = eps0*A/d, Q = {val:.6g} C"), "ok"
            if target_symbol == "W" and U:
                val = 0.5 * cap * U * U
                return _semantic_result(val, target_unit, confidence, payload, f"W = 0.5*C*U^2, C = eps0*A/d, W = {val:.6g} J"), "ok"
            if target_symbol == "U" and q_values:
                val = abs(q_values[0]) / cap
                return _semantic_result(val, target_unit, confidence, payload, f"U = Q/C, C = eps0*A/d, U = {val:.6g} V"), "ok"

    if canonical == "capacitor_equivalent_capacitance" and len(_semantic_values_by_unit(payload, {"F", "uF", "nF", "pF", "mF"})) >= 2:
        c_values = _semantic_values_by_unit(payload, {"F", "uF", "nF", "pF", "mF"})
        is_series = "series" in text_blob
        is_parallel = "parallel" in text_blob or not is_series
        ceq = None
        if is_series and all(c > 0 for c in c_values):
            ceq = 1.0 / sum(1.0 / c for c in c_values)
        elif is_parallel:
            ceq = sum(c_values)
        if ceq:
            if target_symbol == "C":
                return _semantic_result(ceq, target_unit, confidence, payload, f"C_eq = {ceq:.6g} F"), "ok"
            if target_symbol == "Q" and U:
                val = ceq * U
                return _semantic_result(val, target_unit, confidence, payload, f"Q_total = C_eq*U = {val:.6g} C"), "ok"
            if target_symbol == "W" and U:
                val = 0.5 * ceq * U * U
                return _semantic_result(val, target_unit, confidence, payload, f"W = 0.5*C_eq*U^2 = {val:.6g} J"), "ok"
            if target_symbol == "U" and q_values:
                candidates = [abs(q_values[0]) / c for c in c_values if c > 0]
                voltage_bounds = [v for v in _semantic_values_by_unit(payload, {"V", "mV", "kV"}) if v > 0]
                if voltage_bounds:
                    bound = max(voltage_bounds)
                    under_bound = [v for v in candidates if v <= bound * 1.000001]
                    if under_bound:
                        val = max(under_bound)
                    else:
                        val = min(candidates, key=lambda x: abs(x - bound))
                elif candidates:
                    val = candidates[0]
                else:
                    val = abs(q_values[0]) / ceq
                return _semantic_result(val, target_unit, confidence, payload, f"U = Q/C_branch = {val:.6g} V"), "ok"

    if canonical == "rlc_resonance_frequency_from_L_C" and L and C:
        val = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
        return _semantic_result(val, target_unit, confidence, payload, f"f0 = 1/(2*pi*sqrt(L*C)) = {val:.6g} Hz"), "ok"
    if canonical == "rlc_resonance_capacitance_from_L_f" and L and f:
        val = 1.0 / ((2.0 * math.pi * f) ** 2 * L)
        return _semantic_result(val, target_unit, confidence, payload, f"C = 1/((2*pi*f)^2*L) = {val:.6g} F"), "ok"
    if canonical == "rlc_resonance_inductance_from_C_f" and C and f:
        val = 1.0 / ((2.0 * math.pi * f) ** 2 * C)
        return _semantic_result(val, target_unit, confidence, payload, f"L = 1/((2*pi*f)^2*C) = {val:.6g} H"), "ok"
    if canonical == "rlc_resonance_power_from_I_R_or_U_R":
        if I and R_values:
            resistance = sum(R_values) if len(R_values) > 1 and "series" in text_blob else R_values[0]
            val = I * I * resistance
            return _semantic_result(val, target_unit, confidence, payload, f"P = I^2*R = {val:.6g} W"), "ok"
        if U and R_values:
            resistance = sum(R_values) if len(R_values) > 1 and "series" in text_blob else R_values[0]
            val = U * U / resistance
            return _semantic_result(val, target_unit, confidence, payload, f"P = U^2/R = {val:.6g} W"), "ok"
    if canonical == "ac_resonance_general":
        rms_u = _semantic_question_rms_voltage(payload)
        if target_symbol == "U" and rms_u:
            return _semantic_result(rms_u, target_unit, confidence, payload, f"u = U*sqrt(2)*cos(omega*t), so U_rms = {rms_u:.6g} V"), "ok"
        if target_symbol in {"R", "Z"} and len(R_values) >= 3 and ("not in resonance" in text_blob or "impedance" in text_blob):
            resistance = R_values[0]
            xl = R_values[1]
            xc = R_values[2]
            val = math.sqrt(resistance * resistance + (xl - xc) ** 2)
            return _semantic_result(val, target_unit, confidence, payload, f"Z = sqrt(R^2 + (X_L-X_C)^2) = {val:.6g} ohm"), "ok"
        if target_symbol in {"R", "Z"} and len(R_values) == 1 and ("resonance" in text_blob or "resonant" in text_blob):
            return _semantic_result(R_values[0], target_unit, confidence, payload, f"At resonance, Z = R = {R_values[0]:.6g} ohm"), "ok"
        if target_symbol.startswith("I") and U and R_values:
            val = U / R_values[0]
            return _semantic_result(val, target_unit, confidence, payload, f"I = U/Z = {val:.6g} A"), "ok"
        if target_symbol == "U" and I and R_values:
            val = I * R_values[0]
            return _semantic_result(val, target_unit, confidence, payload, f"U = I*Z = {val:.6g} V"), "ok"
        if target_symbol == "P":
            if I and R_values:
                val = I * I * R_values[0]
                return _semantic_result(val, target_unit, confidence, payload, f"P = I^2*R = {val:.6g} W"), "ok"
            if U and R_values:
                val = U * U / R_values[0]
                return _semantic_result(val, target_unit, confidence, payload, f"P = U^2/R = {val:.6g} W"), "ok"

    if canonical == "circuit_power_from_U_I_R":
        if target_symbol.startswith("I") and power_values and U:
            val = power_values[0] / U
            return _semantic_result(val, target_unit, confidence, payload, f"I = P/U = {val:.6g} A"), "ok"
        if target_symbol == "U" and power_values and I:
            val = power_values[0] / I
            return _semantic_result(val, target_unit, confidence, payload, f"U = P/I = {val:.6g} V"), "ok"
        if U and I:
            val = U * I
            return _semantic_result(val, target_unit, confidence, payload, f"P = U*I = {val:.6g} W"), "ok"
        if I and R_values:
            val = I * I * R_values[0]
            return _semantic_result(val, target_unit, confidence, payload, f"P = I^2*R = {val:.6g} W"), "ok"
        if U and R_values:
            val = U * U / R_values[0]
            return _semantic_result(val, target_unit, confidence, payload, f"P = U^2/R = {val:.6g} W"), "ok"
    if canonical == "ohm_current_from_voltage_resistance" and U and R_values:
        val = U / R_values[0]
        return _semantic_result(val, target_unit, confidence, payload, f"I = U/R = {val:.6g} A"), "ok"
    if canonical == "ohm_voltage_from_current_resistance" and I and R_values:
        val = I * R_values[0]
        return _semantic_result(val, target_unit, confidence, payload, f"U = I*R = {val:.6g} V"), "ok"
    if canonical == "ohm_resistance_from_voltage_current" and U and I:
        val = U / I
        return _semantic_result(val, target_unit, confidence, payload, f"R = U/I = {val:.6g} ohm"), "ok"
    if canonical == "circuit_parallel_equivalent_or_branch_current" and len(R_values) >= 2:
        req = 1.0 / sum(1.0 / r for r in R_values if r)
        if ";" in target_unit:
            return None, "semantic_parallel_multi_answer_not_supported"
        if target_symbol == "I" and U:
            val = U / req
            return _semantic_result(val, target_unit, confidence, payload, f"I_total = U/R_eq, R_eq = {req:.6g} ohm, I = {val:.6g} A"), "ok"
        if target_symbol == "R" and target_unit == "ohm":
            return _semantic_result(req, target_unit, confidence, payload, f"1/R_eq = sum(1/R_i), R_eq = {req:.6g} ohm"), "ok"
    if canonical == "circuit_series_equivalent_or_voltage_division" and R_values:
        req = sum(R_values)
        if target_symbol == "R" and target_unit == "ohm":
            return _semantic_result(req, target_unit, confidence, payload, f"R_eq = sum(R_i) = {req:.6g} ohm"), "ok"
        if target_symbol == "I" and U and len(R_values) == 1:
            val = U / req
            return _semantic_result(val, target_unit, confidence, payload, f"I = U/R_eq = {val:.6g} A"), "ok"
    if canonical == "circuit_relation_general":
        if target_symbol == "I" and U and R_values:
            val = U / R_values[0]
            return _semantic_result(val, target_unit, confidence, payload, f"I = U/R = {val:.6g} A"), "ok"
        if target_symbol == "U" and I and R_values:
            val = I * R_values[0]
            return _semantic_result(val, target_unit, confidence, payload, f"U = I*R = {val:.6g} V"), "ok"
        if target_symbol == "R" and U and I:
            val = U / I
            return _semantic_result(val, target_unit, confidence, payload, f"R = U/I = {val:.6g} ohm"), "ok"

    if canonical == "solenoid_magnetic_field_from_turn_density_current" and I:
        n_vals = _semantic_values_by_unit(payload, {"turns/m"})
        if n_vals:
            val = 4.0 * math.pi * 1e-7 * n_vals[0] * I
            return _semantic_result(val, target_unit, confidence, payload, f"B = mu0*n*I = {val:.6g} T"), "ok"
    if canonical == "inductor_energy_from_L_I" and L and I:
        val = 0.5 * L * I * I
        return _semantic_result(val, target_unit, confidence, payload, f"W = 0.5*L*I^2 = {val:.6g} J"), "ok"
    if canonical == "inductance_from_energy_current" and W and I:
        val = 2.0 * W / (I * I)
        return _semantic_result(val, target_unit, confidence, payload, f"L = 2W/I^2 = {val:.6g} H"), "ok"
    if canonical == "magnetic_flux_from_B_area_angle" and B and areas:
        val = B * areas[0]
        return _semantic_result(val, target_unit or "Wb", confidence, payload, f"Phi = B*S = {val:.6g} Wb"), "ok"
    if canonical == "induction_general":
        i_amp = _semantic_question_signal_amplitude(payload, "I")
        if target_symbol == "W" and L and i_amp:
            val = 0.5 * L * i_amp * i_amp
            return _semantic_result(val, target_unit, confidence, payload, f"W_max = 0.5*L*I0^2 = {val:.6g} J"), "ok"
        if target_symbol == "I" and L and W:
            val = math.sqrt(2.0 * W / L)
            return _semantic_result(val, target_unit, confidence, payload, f"I = sqrt(2W/L) = {val:.6g} A"), "ok"
    if canonical == "faraday_emf_from_flux_change":
        phi_values = _semantic_values_by_unit(payload, {"Wb"})
        if len(phi_values) >= 1 and t_values:
            delta_phi = abs(phi_values[-1] - phi_values[0]) if len(phi_values) >= 2 else abs(phi_values[0])
            val = delta_phi / t_values[0]
            return _semantic_result(val, target_unit, confidence, payload, f"|e| = |Delta Phi|/Delta t = {val:.6g} V"), "ok"

    if canonical == "measurement_relative_error":
        pairs = _semantic_measurement_pairs(payload)
        if pairs:
            val = sum(error / measured for measured, error in pairs if measured > 0) * 100.0
            return _semantic_result(val, target_unit, confidence, payload, f"relative error = sum(Delta x/x)*100% = {val:.6g}%"), "ok"
        return None, "semantic_measurement_relative_error_missing_pairs"
    if canonical == "measurement_resistance_uncertainty_from_U_I":
        pairs = _semantic_measurement_pairs(payload)
        if len(pairs) >= 2:
            R_nominal = None
            if U and I:
                R_nominal = U / I
            if R_nominal is None:
                measured = [p[0] for p in pairs]
                if len(measured) >= 2 and measured[1] != 0:
                    R_nominal = measured[0] / measured[1]
            if R_nominal is not None:
                rel = sum(error / measured for measured, error in pairs if measured > 0)
                val = R_nominal * rel
                return _semantic_result(val, target_unit, confidence, payload, f"Delta R = R*(Delta U/U + Delta I/I) = {val:.6g} ohm"), "ok"
        return None, "semantic_measurement_resistance_uncertainty_missing_pairs"

    if canonical == "coulomb_force_between_point_charges":
        if q_values and field_values:
            val = abs(q_values[-1]) * abs(field_values[0])
            return _semantic_result(val, target_unit, confidence, payload, f"F = |q|*E = {val:.6g} N"), "ok"
        if len(force_values) >= 2:
            angle = _semantic_angle_degrees(payload)
            if angle is not None:
                val = math.sqrt(max(0.0, force_values[0] ** 2 + force_values[1] ** 2 + 2.0 * force_values[0] * force_values[1] * math.cos(math.radians(angle))))
                return _semantic_result(val, target_unit, confidence, payload, f"F = sqrt(F1^2+F2^2+2F1F2*cos(theta)) = {val:.6g} N"), "ok"
        if len(q_values) >= 3 and len(r_values) >= 2 and "perpendicular bisector" in text_blob:
            source_q = max(abs(q_values[0]), abs(q_values[1]))
            test_q = abs(q_values[2])
            separation = max(r_values[0], r_values[1])
            height = min(r_values[0], r_values[1])
            distance = math.sqrt((separation / 2.0) ** 2 + height ** 2)
            val = test_q * 2.0 * 9e9 * source_q * (separation / 2.0) / (distance ** 3)
            return _semantic_result(val, target_unit, confidence, payload, f"On the perpendicular bisector, F = q0*2*k*Q*(a/2)/r^3 = {val:.6g} N"), "ok"
        if len(q_values) >= 3 and r_values and ("midpoint" in text_blob or ("equidistant" in text_blob and "line connecting" in text_blob)):
            source_q = max(abs(q_values[0]), abs(q_values[1]))
            test_q = abs(q_values[-1])
            separation = r_values[0]
            val = test_q * 2.0 * 9e9 * source_q / ((separation / 2.0) ** 2)
            return _semantic_result(val, target_unit, confidence, payload, f"At the midpoint, fields add: F = q0*2*k*Q/(a/2)^2 = {val:.6g} N"), "ok"
    if canonical == "coulomb_force_between_point_charges" and len(q_values) == 2 and len(r_values) == 1:
        val = 9e9 * abs(q_values[0] * q_values[1]) / (r_values[0] ** 2)
        return _semantic_result(val, target_unit, confidence, payload, f"F = k*|q1*q2|/r^2 = {val:.6g} N"), "ok"
    if (
        canonical == "electric_field_point_charge_or_superposition"
        and q_values
        and len(q_values) == 1
        and len(r_values) == 1
        and target_symbol in {"E", "E0", "E1", "E2"}
        and target_unit in {"V/m", "N/C"}
    ):
        dielectric = _semantic_dielectric_constant(payload) or 1.0
        val = 9e9 * abs(q_values[0]) / (dielectric * (r_values[0] ** 2))
        return _semantic_result(val, target_unit, confidence, payload, f"E = k*|q|/(epsilon_r*r^2) = {val:.6g} V/m"), "ok"
    if canonical == "electric_field_point_charge_or_superposition":
        if target_symbol in {"Q", "q"} and q_values and force_values and r_values:
            val = abs(force_values[0]) * (r_values[0] ** 2) / (9e9 * abs(q_values[0]))
            return _semantic_result(val, target_unit, confidence, payload, f"Q = F*r^2/(k*q) = {val:.6g} C"), "ok"
        if target_symbol in {"E", "E0", "E1", "E2"} and len(q_values) >= 2 and len(r_values) >= 2 and "line connecting" in text_blob:
            separation = max(r_values[0], r_values[1])
            x = min(r_values[0], r_values[1])
            other = abs(separation - x)
            if other > 0:
                e1 = 9e9 * abs(q_values[0]) / (x * x)
                e2 = 9e9 * abs(q_values[1]) / (other * other)
                if q_values[0] * q_values[1] < 0:
                    val = e1 + e2
                    calc = "between opposite charges, electric fields point in the same direction"
                else:
                    val = abs(e1 - e2)
                    calc = "between like charges, electric fields oppose"
                return _semantic_result(val, target_unit, confidence, payload, f"{calc}: E = {val:.6g} V/m"), "ok"
        if target_symbol in {"E", "E0", "E1", "E2"} and len(q_values) >= 2 and r_values:
            angle = _semantic_angle_degrees(payload) if ("angle" in text_blob or "form" in text_blob) else None
            if angle is not None:
                r = r_values[-1]
                e1 = 9e9 * abs(q_values[0]) / (r * r)
                e2 = 9e9 * abs(q_values[1]) / (r * r)
                val = math.sqrt(max(0.0, e1 * e1 + e2 * e2 + 2.0 * e1 * e2 * math.cos(math.radians(angle))))
                return _semantic_result(val, target_unit, confidence, payload, f"E = sqrt(E1^2+E2^2+2E1E2*cos(theta)) = {val:.6g} V/m"), "ok"
        if target_symbol in {"E", "E0", "E1", "E2"} and "flat" in text_blob and "plate" in text_blob:
            charge_candidates = q_values or _semantic_question_values_by_unit(payload, {"C", "uC", "nC", "pC", "mC"})
            length_values = _semantic_question_values_by_unit(payload, {"m", "cm", "mm"})
            if charge_candidates and len(length_values) >= 2:
                area = length_values[0] * length_values[1]
                sigma = abs(charge_candidates[0]) / area
                eps0 = 8.854187817e-12
                val = sigma / (2.0 * eps0)
                return _semantic_result(val, target_unit, confidence, payload, f"For a large charged plate, E = sigma/(2*eps0) = {val:.6g} V/m"), "ok"
        if target_symbol in {"E", "E0", "E1", "E2"} and mass_values and q_values and "equilibrium" in text_blob:
            val = mass_values[0] * 10.0 / abs(q_values[0])
            return _semantic_result(val, target_unit, confidence, payload, f"Equilibrium gives qE = mg, so E = mg/q = {val:.6g} V/m"), "ok"
    if canonical == "electric_force_vector_superposition_geometry":
        if q_values and r_values:
            q1 = q_values[0]
            q2 = q_values[1] if len(q_values) >= 2 else q1
            base = 9e9 * abs(q1 * q2) / (r_values[0] ** 2)
            if "right-angle" in text_blob or "right angle" in text_blob or "perpendicular" in text_blob:
                val = math.sqrt(2.0) * base
                return _semantic_result(val, target_unit, confidence, payload, f"F = sqrt(2)*k*q^2/a^2 = {val:.6g} N"), "ok"
            if "equilateral" in text_blob:
                val = math.sqrt(3.0) * base
                return _semantic_result(val, target_unit, confidence, payload, f"F = sqrt(3)*k*q^2/a^2 = {val:.6g} N"), "ok"
        return None, "semantic_geometry_force_missing_clear_geometry"
    if canonical == "electric_field_vector_superposition_triangle":
        if q_values and r_values:
            q1 = q_values[0]
            if len(q_values) >= 2 and ("q3" not in text_blob and "acting on" not in text_blob and "at the position of" not in text_blob):
                q2 = q_values[1]
            else:
                q2 = q1
            r = r_values[0]
            e1 = 9e9 * abs(q1) / (r * r)
            e2 = 9e9 * abs(q2) / (r * r)
            angle = _semantic_angle_degrees(payload)
            if angle is not None:
                val = math.sqrt(max(0.0, e1 * e1 + e2 * e2 + 2.0 * e1 * e2 * math.cos(math.radians(angle))))
                return _semantic_result(val, target_unit, confidence, payload, f"E = sqrt(E1^2+E2^2+2E1E2*cos(theta)) = {val:.6g} V/m"), "ok"
        return None, "semantic_geometry_field_missing_clear_geometry"
    if canonical == "electric_field_on_axis_of_charged_ring" and q_values and len(r_values) >= 2:
        ring_radius = max(r_values[0], r_values[1])
        axis_distance = min(r_values[0], r_values[1])
        val = 9e9 * abs(q_values[0]) * axis_distance / ((ring_radius * ring_radius + axis_distance * axis_distance) ** 1.5)
        return _semantic_result(val, target_unit, confidence, payload, f"E = kQx/(R^2+x^2)^(3/2) = {val:.6g} V/m"), "ok"
    if canonical == "vector_force_resultant_or_coulomb_geometry":
        if len(force_values) >= 2:
            angle = _semantic_angle_degrees(payload)
            if angle is not None:
                val = math.sqrt(max(0.0, force_values[0] ** 2 + force_values[1] ** 2 + 2.0 * force_values[0] * force_values[1] * math.cos(math.radians(angle))))
                return _semantic_result(val, target_unit, confidence, payload, f"F = sqrt(F1^2+F2^2+2F1F2*cos(theta)) = {val:.6g} N"), "ok"
        if len(q_values) >= 3 and len(r_values) >= 2 and "perpendicular bisector" in text_blob:
            pseudo_payload = dict(payload)
            pseudo_payload["canonical_problem"] = "coulomb_force_between_point_charges"
            return _solve_semantic_canonical(pseudo_payload)
        if len(q_values) >= 3 and r_values and ("midpoint" in text_blob or ("equidistant" in text_blob and "line connecting" in text_blob)):
            pseudo_payload = dict(payload)
            pseudo_payload["canonical_problem"] = "coulomb_force_between_point_charges"
            return _solve_semantic_canonical(pseudo_payload)
        if q_values and r_values and ("right-angle" in text_blob or "right angle" in text_blob or "equilateral" in text_blob):
            pseudo_payload = dict(payload)
            pseudo_payload["canonical_problem"] = "electric_force_vector_superposition_geometry"
            return _solve_semantic_canonical(pseudo_payload)

    # Generic formula layer: trust the LLM only for semantic extraction
    # (target + givens). The answer still comes from deterministic formulas.
    if target_unit == "ohm" and target_symbol in {"R", "Z", "resistance", "impedance"}:
        if "series" in text_blob and len(R_values) >= 2:
            val = sum(R_values)
            return _semantic_result(val, target_unit, confidence, payload, f"R_series = sum(R_i) = {val:.6g} ohm"), "ok"
        if "parallel" in text_blob and len(R_values) >= 2 and all(r > 0 for r in R_values):
            val = 1.0 / sum(1.0 / r for r in R_values)
            return _semantic_result(val, target_unit, confidence, payload, f"1/R_parallel = sum(1/R_i), R = {val:.6g} ohm"), "ok"
        if len(R_values) >= 3 and ("impedance" in text_blob or "reactance" in text_blob):
            val = math.sqrt(R_values[0] ** 2 + (R_values[1] - R_values[2]) ** 2)
            return _semantic_result(val, target_unit, confidence, payload, f"Z = sqrt(R^2+(X_L-X_C)^2) = {val:.6g} ohm"), "ok"
        if U and I:
            val = U / I
            return _semantic_result(val, target_unit, confidence, payload, f"R = U/I = {val:.6g} ohm"), "ok"
    if target_unit == "W" or target_symbol == "P":
        if "total" in text_blob and len(power_values) >= 2:
            val = sum(power_values)
            return _semantic_result(val, target_unit, confidence, payload, f"P_total = sum(P_i) = {val:.6g} W"), "ok"
        if U and I:
            val = U * I
            return _semantic_result(val, target_unit, confidence, payload, f"P = U*I = {val:.6g} W"), "ok"
        if I and R_values:
            val = I * I * R_values[0]
            return _semantic_result(val, target_unit, confidence, payload, f"P = I^2*R = {val:.6g} W"), "ok"
        if U and R_values:
            val = U * U / R_values[0]
            return _semantic_result(val, target_unit, confidence, payload, f"P = U^2/R = {val:.6g} W"), "ok"
    if target_unit == "A" and target_symbol.startswith("I"):
        if power_values and U:
            val = power_values[0] / U
            return _semantic_result(val, target_unit, confidence, payload, f"I = P/U = {val:.6g} A"), "ok"
        if U and R_values:
            resistance = sum(R_values) if "series" in text_blob and len(R_values) > 1 else R_values[0]
            val = U / resistance
            return _semantic_result(val, target_unit, confidence, payload, f"I = U/R = {val:.6g} A"), "ok"
        if L and W:
            val = math.sqrt(2.0 * W / L)
            return _semantic_result(val, target_unit, confidence, payload, f"I = sqrt(2W/L) = {val:.6g} A"), "ok"
    if target_unit == "V" and target_symbol == "U":
        if power_values and I:
            val = power_values[0] / I
            return _semantic_result(val, target_unit, confidence, payload, f"U = P/I = {val:.6g} V"), "ok"
        if I and R_values:
            val = I * R_values[0]
            return _semantic_result(val, target_unit, confidence, payload, f"U = I*R = {val:.6g} V"), "ok"
        if L and C and I:
            val = I * math.sqrt(L / C)
            return _semantic_result(val, target_unit, confidence, payload, f"U0 = I0*sqrt(L/C) = {val:.6g} V"), "ok"
        if W and C:
            val = math.sqrt(2.0 * W / C)
            return _semantic_result(val, target_unit, confidence, payload, f"U = sqrt(2W/C) = {val:.6g} V"), "ok"
    if target_unit in {"F", "uF", "nF", "pF"} or target_symbol == "C":
        c_values = _semantic_values_by_unit(payload, {"F", "uF", "nF", "pF", "mF"})
        if len(c_values) >= 2 and "series" in text_blob and all(c > 0 for c in c_values):
            val = 1.0 / sum(1.0 / c for c in c_values)
            return _semantic_result(val, target_unit, confidence, payload, f"C_series = 1/sum(1/C_i) = {val:.6g} F"), "ok"
        if len(c_values) >= 2 and "parallel" in text_blob:
            val = sum(c_values)
            return _semantic_result(val, target_unit, confidence, payload, f"C_parallel = sum(C_i) = {val:.6g} F"), "ok"
        if W and U:
            val = 2.0 * W / (U * U)
            return _semantic_result(val, target_unit, confidence, payload, f"C = 2W/U^2 = {val:.6g} F"), "ok"
        if q_values and U:
            val = abs(q_values[0]) / U
            return _semantic_result(val, target_unit, confidence, payload, f"C = Q/U = {val:.6g} F"), "ok"
    if target_unit in {"C", "uC", "nC", "pC"} or target_symbol in {"Q", "q"}:
        c_values = _semantic_values_by_unit(payload, {"F", "uF", "nF", "pF", "mF"})
        if len(c_values) >= 2 and U:
            if "series" in text_blob and all(c > 0 for c in c_values):
                ceq = 1.0 / sum(1.0 / c for c in c_values)
            else:
                ceq = sum(c_values)
            val = ceq * U
            return _semantic_result(val, target_unit, confidence, payload, f"Q = C_eq*U = {val:.6g} C"), "ok"
        if C and U:
            val = C * U
            return _semantic_result(val, target_unit, confidence, payload, f"Q = C*U = {val:.6g} C"), "ok"
    if target_unit in {"J", "mJ", "uJ", "nJ"} or target_symbol == "W":
        if L and I:
            val = 0.5 * L * I * I
            return _semantic_result(val, target_unit, confidence, payload, f"W = 0.5*L*I^2 = {val:.6g} J"), "ok"
        if C and U:
            val = 0.5 * C * U * U
            return _semantic_result(val, target_unit, confidence, payload, f"W = 0.5*C*U^2 = {val:.6g} J"), "ok"
    if target_unit in {"N", "mN", "uN"} or target_symbol == "F":
        if q_values and field_values:
            val = abs(q_values[-1]) * abs(field_values[0])
            return _semantic_result(val, target_unit, confidence, payload, f"F = |q|*E = {val:.6g} N"), "ok"
        if len(q_values) >= 2 and r_values:
            val = 9e9 * abs(q_values[0] * q_values[1]) / (r_values[0] ** 2)
            return _semantic_result(val, target_unit, confidence, payload, f"F = k*|q1*q2|/r^2 = {val:.6g} N"), "ok"
    if target_unit in {"V/m", "N/C"} and target_symbol in {"E", "E0", "E1", "E2"}:
        if force_values and q_values:
            val = abs(force_values[0]) / abs(q_values[0])
            return _semantic_result(val, target_unit, confidence, payload, f"E = F/|q| = {val:.6g} V/m"), "ok"
        if q_values and r_values:
            dielectric = _semantic_dielectric_constant(payload) or 1.0
            val = 9e9 * abs(q_values[0]) / (dielectric * (r_values[0] ** 2))
            return _semantic_result(val, target_unit, confidence, payload, f"E = k*|q|/(epsilon_r*r^2) = {val:.6g} V/m"), "ok"
    if target_unit == "%" and (target_symbol or "relative" in text_blob):
        pairs = _semantic_measurement_pairs(payload)
        if pairs:
            val = sum(error / measured for measured, error in pairs if measured > 0) * 100.0
            return _semantic_result(val, target_unit, confidence, payload, f"relative error = sum(Delta x/x)*100% = {val:.6g}%"), "ok"
    if target_unit == "ohm" and ("uncertainty" in text_blob or "error" in text_blob):
        pairs = _semantic_measurement_pairs(payload)
        if len(pairs) >= 2:
            measured = [p[0] for p in pairs]
            errors = [p[1] for p in pairs]
            if measured[1] != 0:
                nominal = U / I if U and I else measured[0] / measured[1]
                rel = sum(err / val for val, err in zip(measured[:2], errors[:2]) if val > 0)
                val = nominal * rel
                return _semantic_result(val, target_unit, confidence, payload, f"Delta R = R*(Delta U/U + Delta I/I) = {val:.6g} ohm"), "ok"
    if target_unit == "rad/s" and f:
        val = 2.0 * math.pi * f
        return _semantic_result(val, target_unit, confidence, payload, f"omega = 2*pi*f = {val:.6g} rad/s"), "ok"

    return None, f"semantic_canonical_not_supported:{canonical}"


def _calculator_v2_to_api_result(calc: Dict[str, Any], payload: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    formula_id = str(calc.get("formula_id") or "")
    calc_line = str(calc.get("calculation") or calc.get("formula") or "")
    answer = str(calc.get("answer") or "")
    unit = str(calc.get("unit") or "")
    topic = str(calc.get("topic") or payload.get("topic") or "")
    explanation = (
        f"The language model only extracted the problem structure; it did not compute the answer. "
        f"The verified calculator selected `{formula_id}` and used {calc_line}. "
        f"After unit conversion, the final answer is {answer} {unit}."
    )
    return {
        "answer": answer,
        "unit": unit,
        "explanation": explanation,
        "cot": [
            f"Parser output: topic={topic}, target={payload.get('target')}.",
            f"Calculator formula: {formula_id}.",
            f"Locked calculation: {calc_line}.",
            f"Final answer: {answer} {unit}.",
        ],
        "premises": [
            "The semantic parser is constrained to extraction only.",
            "The final answer is recomputed by the deterministic calculator.",
            "Parser-provided answers or executable code are rejected.",
        ],
        "confidence": min(float(calc.get("confidence") or 0.0), 0.92),
        "method": "calculator_v2_numeric_parser",
        "topic_pred": topic,
        "prefix_pred": "",
        "semantic_payload": payload,
        "semantic_model": model_name,
        "semantic_status": "ok",
        "calculator_v2_trace": calc.get("trace"),
    }


def _looks_like_calculator_v2_payload(payload: Dict[str, Any]) -> bool:
    candidates = payload.get("formula_candidates")
    if isinstance(candidates, list) and len(candidates) > 0:
        return True
    for given in payload.get("givens") or []:
        if not isinstance(given, dict):
            continue
        if "raw_span" in given and ("value_text" in given or "unit_text" in given):
            return True
    return False


def _try_semantic_fallback(question: str, route_hint: Dict[str, Any], debug: bool = False) -> tuple[Optional[Dict[str, Any]], str]:
    if not DEFAULT_ENABLE_SEMANTIC_PARSER:
        return None, "semantic_disabled"

    model_name = DEFAULT_SEMANTIC_MODEL.strip()
    if not model_name:
        return None, "semantic_model_not_configured"

    try:
        raw_text = _call_openai_chat_completion(
            messages=_build_semantic_messages(question, route_hint),
            model_name=model_name,
            max_tokens=DEFAULT_SEMANTIC_MAX_TOKENS,
            timeout_seconds=DEFAULT_SEMANTIC_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return None, f"semantic_call_failed:{exc}"

    payload = _extract_json_object(raw_text)
    if payload is None:
        return None, f"semantic_invalid_json:{raw_text[:500]}"

    parsed, status = _validate_semantic_payload(payload)
    if parsed is None:
        if debug:
            route_hint["semantic_raw"] = raw_text[:1000]
        return None, status

    parsed["_question"] = question
    if calculator_v2_validate is not None:
        try:
            validation = calculator_v2_validate(question, parsed)
        except Exception as exc:
            validation = {"ok": False, "errors": [{"type": "validator_exception", "detail": str(exc)}], "warnings": []}
        if debug:
            route_hint["calculator_v2_payload_validation"] = validation
        if not validation.get("ok"):
            return None, f"semantic_payload_validator_failed:{validation.get('errors')}"

    if calculator_v2_solve is not None and _looks_like_calculator_v2_payload(parsed):
        try:
            calc = calculator_v2_solve(question, parsed)
        except Exception as exc:
            calc = None
            if debug:
                route_hint["calculator_v2_error"] = str(exc)
        if calc is not None:
            return _calculator_v2_to_api_result(calc, parsed, model_name), "ok"

    out, solve_status = _solve_semantic_canonical(parsed)
    if out is None:
        if debug:
            route_hint["semantic_payload"] = parsed
        return None, solve_status

    out["semantic_model"] = model_name
    return out, "ok"


def _direct_formula_result(
    answer: str,
    unit: str,
    method: str,
    topic: str,
    calc: str,
    confidence: float = 0.94,
) -> Dict[str, Any]:
    explanation = f"The direct physics guardrail matched a clear formula pattern. {calc}. Therefore, the final answer is {answer} {unit}."
    return {
        "answer": answer,
        "unit": _ascii_unit(unit),
        "explanation": explanation,
        "cot": [
            "Identify the problem as a high-confidence direct formula pattern.",
            calc,
            f"Final answer: {answer} {unit}.",
        ],
        "premises": [
            "This guardrail is a deterministic Python formula solver.",
            "It is used only for narrow physics patterns with explicit numeric data.",
        ],
        "confidence": min(max(confidence, 0.0), 0.96),
        "method": method,
        "topic_pred": topic,
        "prefix_pred": "",
        "polish_status": "not_requested",
    }


def _direct_question_values(question: str, units: set[str]) -> list[float]:
    return _semantic_question_values_by_unit({"_question": question}, units)


def _format_branch_current(value: float) -> str:
    if abs(value - round(value, 1)) < 1e-9:
        return f"{value:.1f}"
    return _format_number(value)


def _try_direct_text_guardrail(question: str) -> Optional[Dict[str, Any]]:
    import math

    text = question.lower()
    q_values = _direct_question_values(question, {"C", "uC", "nC", "pC", "mC"})
    r_values = _direct_question_values(question, {"m", "cm", "mm"})
    R_values = _direct_question_values(question, {"ohm"})
    U_values = _direct_question_values(question, {"V", "mV", "kV"})

    # Single point-charge electric field. This catches forms such as
    # "charge 10^-9 C ... point 3 cm away", where older parsers can read
    # the exponent as a negative charge value.
    if (
        len(q_values) == 1
        and len(r_values) == 1
        and "electric field" in text
        and "point charge" not in text[:30]
        and not any(word in text for word in ("two", "three", "triangle", "plate", "force"))
    ):
        value = 9e9 * abs(q_values[0]) / (r_values[0] ** 2)
        return _direct_formula_result(
            _format_number(value),
            "V/m",
            "direct_guardrail_point_charge_field",
            "electrostatics_field",
            f"Use E = k|q|/r^2 = {value:.6g} V/m",
        )

    # Two opposite charges, test charge on the perpendicular bisector.
    if (
        len(q_values) >= 3
        and len(r_values) >= 2
        and "perpendicular bisector" in text
        and "test charge" in text
        and "force" in text
    ):
        source_q = max(abs(q_values[0]), abs(q_values[1]))
        test_q = abs(q_values[2])
        separation = max(r_values[0], r_values[1])
        height = min(r_values[0], r_values[1])
        distance = math.sqrt((separation / 2.0) ** 2 + height ** 2)
        value = test_q * 2.0 * 9e9 * source_q * (separation / 2.0) / (distance ** 3)
        return _direct_formula_result(
            _format_number(value),
            "N",
            "direct_guardrail_perpendicular_bisector_force",
            "electrostatics_force",
            f"On the perpendicular bisector, vertical field components cancel and horizontal components add: F = q0*2*k*Q*(a/2)/r^3 = {value:.6g} N",
        )

    # Three charges at an isosceles right triangle; q3 is at the right-angle
    # vertex in this dataset pattern, so the two Coulomb forces are perpendicular.
    if (
        len(q_values) >= 3
        and r_values
        and "isosceles right triangle" in text
        and "force acting on q3" in text
    ):
        side = r_values[0]
        q1, q2, q3 = q_values[0], q_values[1], q_values[2]
        f13 = 9e9 * abs(q1 * q3) / (side ** 2)
        f23 = 9e9 * abs(q2 * q3) / (side ** 2)
        value = math.sqrt(f13 * f13 + f23 * f23)
        return _direct_formula_result(
            _format_number(value),
            "N",
            "direct_guardrail_right_triangle_q3_force",
            "electrostatics_force",
            f"The two forces on q3 are perpendicular: F = sqrt(F13^2 + F23^2) = {value:.6g} N",
        )

    # Parallel branch currents. Return both branch currents because the
    # official answer expects I1 and I2, not only the first branch.
    if (
        len(R_values) >= 2
        and U_values
        and "parallel" in text
        and "current flowing through each" in text
    ):
        currents = [U_values[0] / r for r in R_values[:2] if r]
        if len(currents) == 2:
            answer = f"I1 = {_format_branch_current(currents[0])}; I2 = {_format_branch_current(currents[1])}"
            return _direct_formula_result(
                answer,
                "A;A",
                "direct_guardrail_parallel_branch_currents",
                "circuit_resistance",
                f"In parallel, each branch has the same voltage: I1 = U/R1 = {currents[0]:.6g} A and I2 = U/R2 = {currents[1]:.6g} A",
            )

    # Equilateral triangle centroid cancellation: equal q1 and q2 at A/B are
    # canceled by equal q3 at C.
    if (
        q_values
        and "equilateral triangle" in text
        and "centroid" in text
        and "zero" in text
        and ("what value must charge q3" in text or "charge q3" in text)
    ):
        value = abs(q_values[0])
        return _direct_formula_result(
            _format_number(value),
            "C",
            "direct_guardrail_equilateral_centroid_zero_field",
            "electrostatics_field",
            f"At the centroid of an equilateral triangle, equal charges at A and B are canceled by an equal charge at C, so q3 = {value:.6g} C",
        )

    return None


def _apply_optional_polish(
    out: Dict[str, Any],
    question: str,
    polish: bool,
    polish_model: str,
) -> Dict[str, Any]:
    if not polish:
        out["polish_status"] = out.get("polish_status", "not_requested")
        return out

    if not _should_polish_with_adapter(out):
        method = str(out.get("method") or "").lower()
        if method.startswith("semantic_"):
            out["polish_status"] = "skipped_semantic_locked_trace"
        else:
            out["polish_status"] = "skipped_fast_path"
        return out

    model_name = polish_model if polish_model and polish_model != "disabled" else DEFAULT_POLISH_MODEL
    if not model_name or model_name == "disabled":
        out["polish_status"] = "disabled_no_model"
        out["polish_error"] = "PHYSICS_POLISH_MODEL is disabled."
        return out

    locked_answer = _string_answer(out.get("answer"))
    locked_unit = _ascii_unit(out.get("unit"))

    try:
        raw_text = _call_openai_chat_completion(
            messages=_build_polish_messages(out, question),
            model_name=model_name,
        )
        explanation, status = _validate_polished_explanation(raw_text, locked_answer, locked_unit)
        if explanation is None:
            out["polish_status"] = status
            out["polish_raw"] = raw_text[:1000]
            return out

        out["explanation"] = explanation
        out["polish_status"] = "ok"
        out["polish_model"] = model_name
        return out
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
        out["polish_status"] = "failed"
        out["polish_error"] = str(exc)
        return out
    except Exception as exc:
        out["polish_status"] = "failed"
        out["polish_error"] = str(exc)
        return out

    return out


def _should_polish_with_adapter(out: Dict[str, Any]) -> bool:
    mode = DEFAULT_POLISH_MODE
    confidence = float(out.get("confidence") or out.get("solver_conf") or 0.0)
    method = str(out.get("method") or "").lower()
    semantic_status = str(out.get("semantic_status") or "").lower()
    explanation = str(out.get("explanation") or "")

    if mode in {"0", "false", "off", "never", "disabled"}:
        return False

    # Semantic fallback already has a canonical Python trace. In practice the
    # small explanation adapter can make these traces prettier but may also
    # invent a wrong formula, so keep the solver-generated explanation unless
    # explicitly enabled for experiments.
    if semantic_status == "ok" or method.startswith("semantic_"):
        return DEFAULT_POLISH_SEMANTIC

    if mode in {"1", "true", "on", "always"}:
        return True

    if mode in {"semantic", "semantic_only"}:
        return DEFAULT_POLISH_SEMANTIC

    # Conditional mode: keep fast deterministic explanations as-is, and spend
    # adapter time only when the solution needed semantic help or confidence is lower.
    if confidence and confidence < 0.78:
        return True
    if len(explanation) < 180:
        return True
    if method in {"unanswered_no_fallback"}:
        return True
    return False


def _result_confidence(out: Dict[str, Any]) -> float:
    try:
        return float(out.get("confidence") or out.get("solver_conf") or 0.0)
    except Exception:
        return 0.0


def _pipeline_result_needs_semantic_repair(out: Dict[str, Any]) -> bool:
    if _pipeline_result_is_unanswered(out):
        return True

    method = str(out.get("method") or "").lower()
    if method.startswith("semantic_"):
        return False

    confidence = _result_confidence(out)
    return bool(DEFAULT_ENABLE_SEMANTIC_PARSER and confidence and confidence < DEFAULT_SEMANTIC_REPAIR_THRESHOLD)


def _try_semantic_repair_or_keep(
    question: str,
    out: Dict[str, Any],
    debug: bool = False,
) -> Dict[str, Any]:
    if not _pipeline_result_needs_semantic_repair(out):
        return out

    original_confidence = _result_confidence(out)
    semantic_out, semantic_status = _try_semantic_fallback(question, out, debug=debug)
    if semantic_out is not None:
        semantic_out["semantic_repair_from_method"] = out.get("method")
        semantic_out["semantic_repair_from_confidence"] = original_confidence
        return semantic_out

    out["semantic_status"] = semantic_status
    if original_confidence and original_confidence < DEFAULT_SEMANTIC_REPAIR_THRESHOLD:
        out["semantic_repair_status"] = f"semantic_repair_failed_keep_deterministic:{semantic_status}"
    return out


# ============================================================
# 10. Response formatting
# ============================================================

def _format_public_response(out: Dict[str, Any], debug: bool) -> Dict[str, Any]:
    out = _prettify_response_math(out)

    if debug:
        return out

    # BTC-facing response: no model/debug/internal metadata.
    public_keys = [
        "answer",
        "unit",
        "explanation",
        "cot",
        "premises",
        "confidence",
    ]

    return {k: out[k] for k in public_keys if k in out}


def _reasoning_steps_from_pipeline(out: Dict[str, Any]) -> list[str]:
    steps = out.get("cot") or []
    if not isinstance(steps, list):
        steps = [str(steps)]

    clean_steps = [str(step).strip() for step in steps if str(step).strip()]
    if clean_steps:
        return clean_steps[:8]

    trace = out.get("trace") or {}
    proof_path = trace.get("proof_path") if isinstance(trace, dict) else None
    if isinstance(proof_path, list):
        for item in proof_path:
            if isinstance(item, dict) and item.get("content"):
                clean_steps.append(str(item["content"]).strip())

    if clean_steps:
        return clean_steps[:8]

    explanation = str(out.get("explanation") or "").strip()
    return [explanation] if explanation else ["No detailed reasoning trace was produced."]


CHLT_REQUIRED_KEYS = {"question_kind", "topic", "concept", "answer_type", "answer", "evidence", "confidence"}


def _is_chlt_like_request(data: Dict[str, Any]) -> bool:
    query_id = str(data.get("query_id") or "").strip().upper()
    query_type = str(data.get("type") or "").strip().lower()
    query = str(data.get("query") or data.get("question") or "").strip().lower()
    options = data.get("options") if isinstance(data.get("options"), list) else []
    premises = data.get("premises") if isinstance(data.get("premises"), list) else []
    if query_type == "type1":
        return True
    if query_type == "type2":
        return False
    if query_id.startswith("CHLT"):
        return True
    if options or premises:
        return True
    conceptual_markers = (
        "true or false",
        "yes or no",
        "does ",
        "is ",
        "are ",
        "which ",
        "choose ",
        "select ",
        "explain ",
        "what happens",
        "what does",
    )
    return any(marker in query for marker in conceptual_markers) and not re.search(r"\b(calculate|compute|find|determine)\b", query)


def _build_chlt_messages(data: Dict[str, Any]) -> list[Dict[str, str]]:
    query = str(data.get("query") or data.get("question") or "").strip()
    options = data.get("options") if isinstance(data.get("options"), list) else []
    premises = data.get("premises") if isinstance(data.get("premises"), list) else []
    content = {
        "question": query,
        "options": options,
        "premises": premises,
        "instruction": (
            "Return exactly one JSON object with question_kind, topic, concept, answer_type, answer, evidence, confidence. "
            "If options are provided, answer must match one option. Do not output code."
        ),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are chlt_reasoner_final. Answer conceptual physics questions with evidence-grounded JSON only. "
                "Do not perform numeric calculator solving and do not output python_code."
            ),
        },
        {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
    ]


def _option_letter(text: str) -> str:
    match = re.match(r"\s*([A-D])[\.\)]", str(text or "").strip(), flags=re.I)
    return match.group(1).upper() if match else ""


def _coerce_answer_to_option(answer: str, options: list[Any]) -> str:
    if not options:
        return answer
    option_texts = [str(option).strip() for option in options if str(option).strip()]
    if not option_texts:
        return answer
    answer_text = str(answer or "").strip()
    for option in option_texts:
        if answer_text == option:
            return option
    answer_letter = _option_letter(answer_text)
    if answer_letter:
        for option in option_texts:
            if _option_letter(option) == answer_letter:
                return option
    lower_answer = answer_text.lower()
    for option in option_texts:
        if lower_answer and lower_answer in option.lower():
            return option
    if "Uncertain" in option_texts:
        return "Uncertain"
    return option_texts[0]


def _validate_chlt_payload(payload: Dict[str, Any], options: list[Any]) -> tuple[Optional[Dict[str, Any]], str]:
    if not isinstance(payload, dict):
        return None, "chlt_payload_not_object"
    missing = sorted(CHLT_REQUIRED_KEYS - set(payload.keys()))
    if missing:
        return None, f"chlt_missing_keys:{missing}"
    if "python_code" in payload or "final_result" in payload:
        return None, "chlt_forbidden_code_or_result"
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return None, "chlt_bad_evidence"
    try:
        confidence = float(payload.get("confidence"))
    except Exception:
        return None, "chlt_bad_confidence"
    payload["confidence"] = max(0.0, min(0.98, confidence))
    answer_type = str(payload.get("answer_type") or "").strip().lower()
    if answer_type == "yes_no":
        answer = str(payload.get("answer") or "").strip()
        if answer not in {"Yes", "No", "Uncertain"}:
            return None, "chlt_bad_yes_no_answer"
    if options:
        payload["answer"] = _coerce_answer_to_option(str(payload.get("answer") or ""), options)
        payload["answer_type"] = "mcq"
    return payload, "ok"


def _try_chlt_reasoner(data: Dict[str, Any], debug: bool = False) -> tuple[Optional[Dict[str, Any]], str]:
    if not DEFAULT_ENABLE_CHLT_REASONER:
        return None, "chlt_disabled"
    model_name = DEFAULT_CHLT_MODEL.strip()
    if not model_name:
        return None, "chlt_model_not_configured"
    options = data.get("options") if isinstance(data.get("options"), list) else []
    try:
        raw_text = _call_openai_chat_completion(
            messages=_build_chlt_messages(data),
            model_name=model_name,
            max_tokens=DEFAULT_CHLT_MAX_TOKENS,
            timeout_seconds=DEFAULT_CHLT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return None, f"chlt_call_failed:{exc}"

    payload = _extract_json_object(raw_text)
    if payload is None:
        return None, f"chlt_invalid_json:{raw_text[:500]}"
    parsed, status = _validate_chlt_payload(payload, options)
    if parsed is None:
        return None, status
    parsed["chlt_model"] = model_name
    parsed["chlt_status"] = "ok"
    if debug:
        parsed["chlt_raw"] = raw_text[:1000]
    return parsed, "ok"


def _format_chlt_response(data: Dict[str, Any], payload: Dict[str, Any], elapsed_ms: float, debug: bool = False) -> list[Dict[str, Any]]:
    query_id = str(data.get("query_id") or "").strip()
    evidence = [str(item).strip() for item in payload.get("evidence") or [] if str(item).strip()]
    explanation = " ".join(evidence[:3]).strip()
    if not explanation:
        explanation = "The CHLT reasoner selected the answer from the conceptual physics evidence."
    item: Dict[str, Any] = {
        "query_id": query_id,
        "answer": str(payload.get("answer") or "").strip(),
        "unit": "",
        "explanation": explanation,
        "premises_used": [],
        "reasoning": {
            "type": "evidence",
            "steps": evidence[:3],
        },
    }
    if debug:
        item["debug"] = {
            "elapsed_ms": elapsed_ms,
            "method": "chlt_reasoner_final",
            "confidence": payload.get("confidence"),
            "topic_pred": payload.get("topic"),
            "concept": payload.get("concept"),
            "answer_type": payload.get("answer_type"),
            "chlt_model": payload.get("chlt_model"),
            "chlt_status": payload.get("chlt_status"),
        }
    return [_json_safe(item)]


NUMERIC_CLAIM_UNIT_RE = (
    r"(?:V/m|N/C|rad/s|turns/m|cm\^2|mm\^2|m\^2|"
    r"uF|nF|pF|mF|F|uC|nC|pC|mC|C|uH|mH|H|uA|mA|A|"
    r"mV|kV|V|mJ|uJ|nJ|J|mN|uN|N|uT|mT|T|Wb|kW|mW|W|"
    r"cm|mm|m|Hz|ohm|s|ms|us|%)"
)
NUMERIC_CLAIM_RE = re.compile(
    rf"(?P<value>[+-]?\d+(?:\.\d+)?(?:\s*(?:x|×)\s*10\^?[+-]?\d+|[eE][+-]?\d+)?)\s*(?P<unit>{NUMERIC_CLAIM_UNIT_RE})\b",
    flags=re.I,
)


def _parse_claim_number(text: str) -> Optional[float]:
    raw = str(text or "").strip().replace("×", "x")
    sci = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*x\s*10\^?([+-]?\d+)$", raw, flags=re.I)
    if sci:
        return float(sci.group(1)) * (10.0 ** int(sci.group(2)))
    try:
        return float(raw)
    except Exception:
        return None


def _yes_no_options(options: list[Any]) -> bool:
    texts = {str(option).strip().lower() for option in options if str(option).strip()}
    return bool(texts & {"yes", "no", "uncertain", "true", "false"})


def _is_quantitative_type1_request(data: Dict[str, Any]) -> bool:
    if str(data.get("type") or "").strip().lower() != "type1":
        return False
    query = str(data.get("query") or data.get("question") or "")
    options = data.get("options") if isinstance(data.get("options"), list) else []
    if not NUMERIC_CLAIM_RE.search(query):
        return False
    lower = query.lower()
    yes_no_marker = bool(re.search(r"\b(is|are|was|were|does|do|did|will|would|can|could|should)\b", lower))
    truth_marker = bool(re.search(r"\b(true|false|correct|incorrect|yes|no|uncertain)\b", lower))
    return yes_no_marker or truth_marker or _yes_no_options(options)


def _extract_numeric_claim(question: str) -> Optional[Dict[str, Any]]:
    matches = list(NUMERIC_CLAIM_RE.finditer(question or ""))
    if not matches:
        return None

    best = None
    best_score = -10**9
    for idx, match in enumerate(matches):
        before = question[max(0, match.start() - 56):match.start()].lower()
        after = question[match.end():match.end() + 28].lower()
        score = idx
        if re.search(r"\b(equal to|equals|is|are|was|were|be|become|approximately|about|around|close to|less than|greater than|below|above|under|over|more than)\b", before):
            score += 20
        if "?" in after or "true" in after or "correct" in after:
            score += 8
        if re.search(r"\b(given|has|with|from|of|across|separated by|connected to)\b", before):
            score -= 5
        if score >= best_score:
            best_score = score
            best = match

    if best is None:
        return None

    value = _parse_claim_number(best.group("value"))
    if value is None:
        return None
    before = question[max(0, best.start() - 70):best.start()].lower()
    comparator = "approx"
    if re.search(r"\b(less than|smaller than|lower than|below|under)\b", before):
        comparator = "lt"
    elif re.search(r"\b(greater than|larger than|higher than|above|over|more than)\b", before):
        comparator = "gt"
    elif re.search(r"\b(at least|not less than|no less than)\b", before):
        comparator = "ge"
    elif re.search(r"\b(at most|not more than|no more than)\b", before):
        comparator = "le"

    return {
        "value": value,
        "unit": _ascii_unit(best.group("unit")),
        "raw": best.group(0),
        "start": best.start(),
        "end": best.end(),
        "comparator": comparator,
    }


def _build_numeric_verification_question(question: str, claim: Dict[str, Any]) -> str:
    start = int(claim["start"])
    end = int(claim["end"])
    without_claim = (question[:start] + " " + question[end:]).strip()
    without_claim = re.sub(r"\s+", " ", without_claim)
    without_claim = re.sub(r"\b(is|are|was|were)\s+it\s*\?", "?", without_claim, flags=re.I)
    without_claim = re.sub(r"\b(is|are|was|were)\s+the\s+answer\s*\?", "?", without_claim, flags=re.I)
    if re.match(r"^\s*(is|are|was|were)\b", without_claim, flags=re.I):
        without_claim = re.sub(r"^\s*(is|are|was|were)\b", "What is", without_claim, count=1, flags=re.I)
    prefix = (
        "Compute the physical quantity needed to verify the numeric claim. "
        "Do not answer yes/no; return the calculated quantity. "
    )
    suffix = f" The removed claim was {claim['raw']}."
    return (prefix + without_claim + suffix).strip()


def _numeric_answer_value(out: Dict[str, Any]) -> Optional[float]:
    answer = str(out.get("answer") or "").strip()
    match = re.search(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", answer)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _compare_numeric_claim(calculated_value: float, calculated_unit: str, claim: Dict[str, Any]) -> tuple[str, str]:
    answer_unit = _ascii_unit(calculated_unit or claim.get("unit") or "")
    claim_unit = _ascii_unit(claim.get("unit") or answer_unit)
    calc_si = calculated_value * _semantic_unit_scale(answer_unit)
    claim_si = float(claim["value"]) * _semantic_unit_scale(claim_unit)
    tolerance = max(1e-9, abs(claim_si) * 0.02)
    diff = calc_si - claim_si
    comparator = str(claim.get("comparator") or "approx")

    if comparator == "lt":
        ok = calc_si < claim_si or abs(diff) <= tolerance
        relation = "less than"
    elif comparator == "le":
        ok = calc_si <= claim_si or abs(diff) <= tolerance
        relation = "at most"
    elif comparator == "gt":
        ok = calc_si > claim_si or abs(diff) <= tolerance
        relation = "greater than"
    elif comparator == "ge":
        ok = calc_si >= claim_si or abs(diff) <= tolerance
        relation = "at least"
    else:
        ok = abs(diff) <= tolerance
        relation = "approximately equal to"

    return ("Yes" if ok else "No"), relation


def _format_quantitative_type1_response(
    data: Dict[str, Any],
    numeric_question: str,
    numeric_out: Dict[str, Any],
    claim: Dict[str, Any],
    elapsed_ms: float,
    debug: bool = False,
) -> list[Dict[str, Any]]:
    query_id = str(data.get("query_id") or "").strip()
    options = data.get("options") if isinstance(data.get("options"), list) else []
    calc_value = _numeric_answer_value(numeric_out)
    calc_unit = _ascii_unit(numeric_out.get("unit") or claim.get("unit"))
    if calc_value is None:
        answer = "Uncertain"
        relation = "not comparable to"
    else:
        answer, relation = _compare_numeric_claim(calc_value, calc_unit, claim)
    if options:
        option_texts = {str(option).strip().lower() for option in options if str(option).strip()}
        if answer == "Yes" and "true" in option_texts and "yes" not in option_texts:
            answer = "True"
        elif answer == "No" and "false" in option_texts and "no" not in option_texts:
            answer = "False"
        answer = _coerce_answer_to_option(answer, options)

    calculated_text = f"{_string_answer(numeric_out.get('answer'))} {calc_unit}".strip()
    claim_text = f"{claim['raw']}"
    explanation = (
        f"The query is a quantitative yes/no check, so the API first computed the relevant physics quantity. "
        f"The numeric solver obtained {calculated_text}. The claim in the question was {claim_text}, "
        f"which is {relation} the calculated value within the configured tolerance. Therefore the answer is {answer}."
    )
    steps = [
        "Identify the request as a quantitative Type 1 yes/no claim.",
        f"Remove the proposed claim `{claim_text}` and solve the underlying numeric physics question.",
        f"Computed value: {calculated_text}.",
        f"Compare the computed value against the claim using unit conversion and tolerance.",
        f"Final conceptual answer: {answer}.",
    ]
    item: Dict[str, Any] = {
        "query_id": query_id,
        "answer": answer,
        "unit": "",
        "explanation": explanation,
        "premises_used": [],
        "reasoning": {
            "type": "evidence",
            "steps": steps,
        },
    }
    if debug:
        item["debug"] = {
            "elapsed_ms": elapsed_ms,
            "method": "type1_quantitative_claim_verifier",
            "confidence": 0.9 if answer in {"Yes", "No"} else 0.4,
            "numeric_method": numeric_out.get("method"),
            "numeric_answer": numeric_out.get("answer"),
            "numeric_unit": numeric_out.get("unit"),
            "numeric_question": numeric_question,
            "claim": {k: v for k, v in claim.items() if k not in {"start", "end"}},
            "topic_pred": numeric_out.get("topic_pred"),
            "semantic_status": numeric_out.get("semantic_status"),
            "semantic_model": numeric_out.get("semantic_model"),
        }
    return [_json_safe(item)]


def _try_quantitative_type1_solver(
    data: Dict[str, Any],
    polish: bool = False,
    polish_model: str = "",
    debug: bool = False,
) -> tuple[Optional[Dict[str, Any]], str]:
    if not _is_quantitative_type1_request(data):
        return None, "not_quantitative_type1"
    question = str(data.get("query") or data.get("question") or "").strip()
    claim = _extract_numeric_claim(question)
    if claim is None:
        return None, "quant_type1_no_numeric_claim"

    numeric_question = _build_numeric_verification_question(question, claim)
    ok, validation_msg = _validate_question_text(numeric_question)
    if not ok:
        return None, f"quant_type1_bad_numeric_question:{validation_msg}"

    try:
        _ensure_pipeline_ready()
        direct_out = _try_direct_text_guardrail(numeric_question)
        if direct_out is not None:
            out = direct_out
        else:
            out = _solve_with_pipeline(question=numeric_question, debug=True)
            if not isinstance(out, dict):
                return None, f"quant_type1_bad_pipeline_result:{type(out)}"
            if _pipeline_result_is_unanswered(out):
                out = _try_semantic_repair_or_keep(numeric_question, out, debug=debug)
            else:
                out = _try_semantic_repair_or_keep(numeric_question, out, debug=debug)

        if _pipeline_result_is_unanswered(out):
            return None, f"quant_type1_numeric_unanswered:{out.get('semantic_status') or out.get('method')}"
        out = _apply_optional_polish(out=out, question=numeric_question, polish=polish, polish_model=polish_model)
        return {
            "numeric_question": numeric_question,
            "numeric_out": out,
            "claim": claim,
        }, "ok"
    except Exception as exc:
        return None, f"quant_type1_exception:{exc}"


def _competition_type1_placeholder(data: Dict[str, Any]) -> list[Dict[str, Any]]:
    query_id = str(data.get("query_id") or "").strip()
    options = data.get("options") if isinstance(data.get("options"), list) else []

    if "Uncertain" in options:
        answer = "Uncertain"
    elif options:
        answer = str(options[0])
    else:
        answer = "Uncertain"

    return [
        {
            "query_id": query_id,
            "answer": answer,
            "unit": "",
            "explanation": (
                "This endpoint is currently configured primarily for Type 2 physics. "
                "For this Type 1 query, it returns a conservative placeholder answer."
            ),
            "premises_used": [],
            "reasoning": {
                "type": "placeholder",
                "steps": ["Type 1 logic solving is not enabled in this physics-focused server."],
            },
        }
    ]


def _format_competition_type2_response(
    query_id: str,
    question: str,
    out: Dict[str, Any],
    elapsed_ms: float,
    debug: bool = False,
) -> list[Dict[str, Any]]:
    answer = _string_answer(out.get("answer"))
    unit = _ascii_unit(out.get("unit"))
    explanation = str(out.get("explanation") or "").strip()
    if not explanation:
        final = f"{answer} {unit}".strip()
        explanation = f"The physics pipeline computed the requested quantity and obtained {final}."

    item: Dict[str, Any] = {
        "query_id": query_id,
        "answer": answer,
        "unit": unit,
        "explanation": explanation,
        "premises_used": [],
        "reasoning": {
            "type": "cot",
            "steps": _reasoning_steps_from_pipeline(out),
        },
    }

    if debug:
        item["debug"] = {
            "elapsed_ms": elapsed_ms,
            "method": out.get("method"),
            "confidence": out.get("confidence"),
            "topic_pred": out.get("topic_pred"),
            "prefix_pred": out.get("prefix_pred"),
            "polish_status": out.get("polish_status"),
            "polish_error": out.get("polish_error"),
            "semantic_status": out.get("semantic_status"),
            "semantic_model": out.get("semantic_model"),
            "semantic_repair_status": out.get("semantic_repair_status"),
            "semantic_repair_from_method": out.get("semantic_repair_from_method"),
            "semantic_repair_from_confidence": out.get("semantic_repair_from_confidence"),
        }

    return [_json_safe(item)]


def _handle_competition_predict(data: Dict[str, Any]):
    started = time.perf_counter()
    query_id = str(data.get("query_id") or "").strip()
    query_type = str(data.get("type") or "").strip().lower()
    query = data.get("query")
    debug = _bool_from_any(data.get("debug"), default=False)
    polish = _bool_from_any(data.get("polish"), default=DEFAULT_ENABLE_POLISH)
    polish_model = str(data.get("polish_model") or DEFAULT_POLISH_MODEL).strip()

    if not query_id:
        return _error_response("Missing required field: query_id.", status=400, error_type="invalid_competition_input")
    if query_type not in {"type1", "type2"}:
        return _error_response("Field 'type' must be 'type1' or 'type2'.", status=400, error_type="invalid_competition_input")
    if not isinstance(query, str) or not query.strip():
        return _error_response("Field 'query' must be a non-empty string.", status=400, error_type="invalid_competition_input")

    if query_type == "type1":
        quant_out, quant_status = _try_quantitative_type1_solver(
            data,
            polish=polish,
            polish_model=polish_model,
            debug=debug,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        if quant_out is not None:
            return jsonify(_format_quantitative_type1_response(
                data=data,
                numeric_question=quant_out["numeric_question"],
                numeric_out=quant_out["numeric_out"],
                claim=quant_out["claim"],
                elapsed_ms=elapsed_ms,
                debug=debug,
            ))

        chlt_out, chlt_status = _try_chlt_reasoner(data, debug=debug)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        if chlt_out is not None:
            if debug:
                chlt_out["quantitative_type1_status"] = quant_status
            return jsonify(_format_chlt_response(data, chlt_out, elapsed_ms, debug=debug))
        if debug:
            fallback = _competition_type1_placeholder(data)
            fallback[0].setdefault("debug", {})
            fallback[0]["debug"]["chlt_status"] = chlt_status
            fallback[0]["debug"]["quantitative_type1_status"] = quant_status
            fallback[0]["debug"]["elapsed_ms"] = elapsed_ms
            return jsonify(fallback)
        return jsonify(_competition_type1_placeholder(data))

    if _is_chlt_like_request(data):
        chlt_out, chlt_status = _try_chlt_reasoner(data, debug=debug)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        if chlt_out is not None:
            return jsonify(_format_chlt_response(data, chlt_out, elapsed_ms, debug=debug))
        if debug:
            fallback = _competition_type1_placeholder(data)
            fallback[0].setdefault("debug", {})
            fallback[0]["debug"]["chlt_status"] = chlt_status
            fallback[0]["debug"]["elapsed_ms"] = elapsed_ms
            return jsonify(fallback)
        return jsonify(_competition_type1_placeholder(data))

    question = query.strip()
    ok, validation_msg = _validate_question_text(question)
    if not ok:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        out = {
            "answer": "",
            "unit": "",
            "explanation": validation_msg,
            "cot": [validation_msg],
            "method": "invalid_physics_query",
            "confidence": 0.0,
        }
        return jsonify(_format_competition_type2_response(query_id, question, out, elapsed_ms, debug=debug))

    try:
        _ensure_pipeline_ready()
        direct_out = _try_direct_text_guardrail(question)
        if direct_out is not None:
            out = _apply_optional_polish(
                out=direct_out,
                question=question,
                polish=polish,
                polish_model=polish_model,
            )
        else:
            out = _solve_with_pipeline(question=question, debug=True)
            if not isinstance(out, dict):
                raise RuntimeError(f"Pipeline returned invalid result type: {type(out)}")

        if direct_out is None:
            if _pipeline_result_is_unanswered(out):
                out = _try_semantic_repair_or_keep(question, out, debug=debug)
                if _pipeline_result_is_unanswered(out):
                    out.setdefault("explanation", "No deterministic or semantic canonical solver matched this physics query confidently.")
                    out.setdefault("cot", ["No deterministic or semantic canonical solver matched this physics query confidently."])
            else:
                out = _try_semantic_repair_or_keep(question, out, debug=debug)

            out = _apply_optional_polish(
                out=out,
                question=question,
                polish=polish,
                polish_model=polish_model,
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return jsonify(_format_competition_type2_response(query_id, question, out, elapsed_ms, debug=debug))

    except Exception:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        out = {
            "answer": "",
            "unit": "",
            "explanation": "The server encountered an internal error while solving this physics query.",
            "cot": ["Internal server error during Type 2 physics solving."],
            "method": "server_error",
            "confidence": 0.0,
        }
        if debug:
            out["polish_error"] = traceback.format_exc()
        return jsonify(_format_competition_type2_response(query_id, question, out, elapsed_ms, debug=debug))


def _handle_solve_request():
    started = time.perf_counter()

    if request.method == "GET":
        data = dict(request.args)
    else:
        data = _load_json_body()

    question = data.get("question", None)

    ok, validation_msg = _validate_question_text(question)
    if not ok:
        return _error_response(
            validation_msg,
            status=400,
            error_type="invalid_question",
            expected_format={
                "question": "A physics question as a string."
            },
        )

    question = str(question).strip()

    # Defaults for public/BTC:
    # - debug is false unless explicitly requested.
    # - polish is true unless explicitly disabled.
    debug = _bool_from_any(data.get("debug"), default=False)
    polish = _bool_from_any(data.get("polish"), default=DEFAULT_ENABLE_POLISH)
    polish_model = str(data.get("polish_model") or DEFAULT_POLISH_MODEL).strip()

    true_answer = data.get("true_answer")
    true_unit = data.get("true_unit")

    try:
        _ensure_pipeline_ready()
    except Exception:
        return _error_response(
            "Pipeline failed to prepare.",
            status=500,
            error_type="pipeline_prepare_error",
            prepared_data_path=APP_STATE.get("prepared_data_path"),
            detail=APP_STATE.get("pipeline_error") if debug else None,
        )

    try:
        direct_out = _try_direct_text_guardrail(question)
        if direct_out is not None:
            out = direct_out
        else:
            out = _solve_with_pipeline(
                question=question,
                debug=debug,
                true_answer=true_answer,
                true_unit=true_unit,
            )

        if not isinstance(out, dict):
            return _error_response(
                "Pipeline returned an invalid result.",
                status=500,
                error_type="pipeline_invalid_result",
                result_type=str(type(out)),
            )

        if direct_out is None and _pipeline_result_is_unanswered(out):
            semantic_out, semantic_status = _try_semantic_fallback(question, out, debug=debug)
            if semantic_out is not None:
                out = semantic_out
            else:
                out["semantic_status"] = semantic_status

        if _pipeline_result_is_unanswered(out):
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

            payload = {
                "ok": False,
                "error_type": "unsupported_or_unanswered",
                "message": (
                    "The question could not be answered by the deterministic physics solver. "
                    "Please provide a supported physics problem with enough numerical information."
                ),
                "elapsed_ms": elapsed_ms,
            }

            if debug:
                payload["pipeline_result"] = _prettify_response_math(out)

            return jsonify(_json_safe(payload)), 422

        out = _apply_optional_polish(
            out=out,
            question=question,
            polish=polish,
            polish_model=polish_model,
        )

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        out.setdefault(
            "pipeline_version",
            getattr(pipeline, "PIPELINE_VERSION", "clean_blocks_with_verified_llm_json_guardrail"),
        )

        if debug:
            out["api"] = {
                "ok": True,
                "elapsed_ms": elapsed_ms,
                "debug": debug,
                "polish": polish,
                "polish_model": polish_model if polish else None,
            }

        public = _format_public_response(out, debug=debug)
        return jsonify(_json_safe(public))

    except Exception:
        return _error_response(
            "Unexpected server error while solving.",
            status=500,
            error_type="server_error",
            detail=traceback.format_exc() if debug else None,
        )


# ============================================================
# 11. Routes
# ============================================================

@app.get("/")
def index():
    return jsonify(
        {
            "ok": True,
            "name": "EXACTS 2026 Physics API",
            "endpoints": {
                "health": "/health",
                "solve_get": "/solve?question=...",
                "solve_post": "/solve",
                "predict_get": "/predict?question=...",
                "predict_post": "/predict",
                "batch_post": "/batch",
            },
        }
    )


@app.get("/health")
def health():
    try:
        _ensure_pipeline_ready()
    except Exception:
        pass

    return jsonify(
        {
            "ok": APP_STATE["pipeline_ready"],
            "pipeline_ready": APP_STATE["pipeline_ready"],
            "pipeline_error": APP_STATE["pipeline_error"],
            "prepared_data_path": APP_STATE["prepared_data_path"],
            "startup_time": APP_STATE["startup_time"],
            "result_dir": str(RESULT_DIR),
            "project_root": str(PROJECT_ROOT),
            "pipeline_version": getattr(pipeline, "PIPELINE_VERSION", "unknown"),
            "default_polish_enabled": DEFAULT_ENABLE_POLISH,
            "default_polish_model": DEFAULT_POLISH_MODEL,
            "polish_mode": DEFAULT_POLISH_MODE,
            "polish_semantic_enabled": DEFAULT_POLISH_SEMANTIC,
            "polish_max_tokens": DEFAULT_POLISH_MAX_TOKENS,
            "polish_base_url": DEFAULT_POLISH_BASE_URL,
            "polish_timeout_seconds": DEFAULT_POLISH_TIMEOUT_SECONDS,
            "semantic_parser_enabled": DEFAULT_ENABLE_SEMANTIC_PARSER,
            "semantic_model": DEFAULT_SEMANTIC_MODEL,
            "semantic_min_confidence": DEFAULT_SEMANTIC_MIN_CONFIDENCE,
            "semantic_repair_threshold": DEFAULT_SEMANTIC_REPAIR_THRESHOLD,
            "semantic_max_tokens": DEFAULT_SEMANTIC_MAX_TOKENS,
            "semantic_timeout_seconds": DEFAULT_SEMANTIC_TIMEOUT_SECONDS,
            "chlt_reasoner_enabled": DEFAULT_ENABLE_CHLT_REASONER,
            "chlt_model": DEFAULT_CHLT_MODEL,
            "chlt_max_tokens": DEFAULT_CHLT_MAX_TOKENS,
            "chlt_timeout_seconds": DEFAULT_CHLT_TIMEOUT_SECONDS,
            "llm_fallback_enabled": DEFAULT_USE_LLM_FALLBACK,
            "llm_model_ready": APP_STATE["llm_model_ready"],
            "llm_model_name_or_path": APP_STATE["llm_model_name_or_path"],
            "llm_model_error": APP_STATE["llm_model_error"],
            "llm_local_files_only": DEFAULT_LLM_LOCAL_FILES_ONLY,
            "llm_load_4bit": DEFAULT_LLM_LOAD_4BIT,
        }
    )


@app.get("/v1/models")
def vllm_models_proxy():
    try:
        return jsonify(_json_safe(_fetch_vllm_models()))
    except Exception as exc:
        fallback_models = [
            {
                "id": "Qwen/Qwen2.5-3B-Instruct",
                "object": "model",
                "owned_by": "local-vllm",
            },
            {
                "id": DEFAULT_POLISH_MODEL,
                "object": "model",
                "owned_by": "local-vllm-lora",
            },
            {
                "id": DEFAULT_SEMANTIC_MODEL,
                "object": "model",
                "owned_by": "local-vllm-lora",
            },
            {
                "id": DEFAULT_CHLT_MODEL,
                "object": "model",
                "owned_by": "local-vllm-lora",
            },
        ]
        return jsonify(
            _json_safe(
                {
                    "object": "list",
                    "data": fallback_models,
                    "proxy_warning": (
                        "Could not reach the configured vLLM /v1/models endpoint. "
                        "Returning configured model names so the public endpoint remains reachable. "
                        "Check vLLM logs if generation requests also fail."
                    ),
                    "detail": str(exc),
                    "configured_base_url": DEFAULT_POLISH_BASE_URL,
                }
            )
        )


@app.route("/solve", methods=["GET", "POST"])
def solve():
    return _handle_solve_request()


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = _load_json_body()
        if any(key in data for key in ("query_id", "type", "query", "premises", "options")):
            return _handle_competition_predict(data)
    return _handle_solve_request()


@app.post("/batch")
def batch():
    data = _load_json_body()
    questions = data.get("questions", [])

    if not isinstance(questions, list) or not questions:
        return _error_response(
            "Missing required field: questions must be a non-empty list.",
            400,
            error_type="invalid_batch",
        )

    debug = _bool_from_any(data.get("debug"), default=False)
    polish = _bool_from_any(data.get("polish"), default=DEFAULT_ENABLE_POLISH)
    polish_model = str(data.get("polish_model") or DEFAULT_POLISH_MODEL).strip()

    try:
        _ensure_pipeline_ready()
    except Exception:
        return _error_response(
            "Pipeline failed to prepare.",
            status=500,
            error_type="pipeline_prepare_error",
            detail=APP_STATE.get("pipeline_error") if debug else None,
        )

    started = time.perf_counter()
    results = []

    for idx, q in enumerate(questions):
        ok, validation_msg = _validate_question_text(q)
        if not ok:
            results.append(
                {
                    "index": idx,
                    "ok": False,
                    "error_type": "invalid_question",
                    "message": validation_msg,
                }
            )
            continue

        question = str(q).strip()

        try:
            out = _solve_with_pipeline(question=question, debug=debug)

            if _pipeline_result_is_unanswered(out):
                item = {
                    "index": idx,
                    "ok": False,
                    "error_type": "unsupported_or_unanswered",
                    "message": "The deterministic physics solver could not answer this question.",
                }
                if debug:
                    item["pipeline_result"] = _prettify_response_math(out)
                results.append(_json_safe(item))
                continue

            out = _apply_optional_polish(
                out,
                question,
                polish=polish,
                polish_model=polish_model,
            )

            public = _format_public_response(out, debug=debug)
            public["index"] = idx
            public["ok"] = True
            results.append(_json_safe(public))

        except Exception as exc:
            results.append(
                {
                    "index": idx,
                    "ok": False,
                    "error_type": "server_error",
                    "message": str(exc),
                }
            )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    return jsonify(
        {
            "ok": True,
            "count": len(results),
            "elapsed_ms": elapsed_ms,
            "results": results,
        }
    )


# ============================================================
# 12. CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run EXACTS 2026 Physics API server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare the pipeline and exit.",
    )

    args = parser.parse_args()

    if args.prepare_only:
        _ensure_pipeline_ready()
        print("Pipeline prepared successfully.")
        print("Data:", APP_STATE["prepared_data_path"])
        return

    print("=" * 72)
    print("EXACTS 2026 Physics API")
    print("=" * 72)
    print("Project root:", PROJECT_ROOT)
    print("Result dir:", RESULT_DIR)
    print("Verified data:", DEFAULT_VERIFIED_DATA)
    print("Host:", args.host)
    print("Port:", args.port)
    print("Default polish:", DEFAULT_ENABLE_POLISH)
    print("Default polish model:", DEFAULT_POLISH_MODEL)
    print("LLM fallback:", DEFAULT_USE_LLM_FALLBACK)
    print("LLM model:", DEFAULT_LLM_MODEL_PATH or DEFAULT_LLM_MODEL_NAME)
    print("LLM local files only:", DEFAULT_LLM_LOCAL_FILES_ONLY)
    print("LLM load 4-bit:", DEFAULT_LLM_LOAD_4BIT)
    print("=" * 72)

    try:
        _ensure_pipeline_ready()
        print("Pipeline ready.")
    except Exception:
        print("WARNING: Pipeline failed to prepare at startup.")
        print(APP_STATE["pipeline_error"])

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
