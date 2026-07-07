from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import z3


ChatFn = Callable[[List[Dict[str, str]], str, Optional[int], Optional[float]], str]


_OPT = re.compile(r"^\s*([A-D])[\.\)]\s*(.+)$", re.I)
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


TYPE1_SYSTEM_PROMPT = """You convert educational logic statements into parser-safe first-order logic.

Return only a JSON list of strings. Use exactly one FOL string per input line.

Allowed syntax:
- ForAll(x, ...)
- Exists(x, ...)
- Not(...)
- Implies(A, B)
- And(A, B)
- Or(A, B)
- Predicate(arg1, arg2, ...)

Rules:
- Use ASCII identifiers only.
- Reuse the same predicate name for the same meaning across all lines.
- Keep implication direction exactly.
- Do not output arithmetic comparisons, True, False, tautologies, markdown, or explanations.
- Treat repeated base-domain nouns as implicit unless they are the actual condition.
- Yes/no questions like "Are all X P?" become ForAll(x, P(x)).
- "Does it follow that ..." means convert only the embedded claim.
- Named facts become predicate applications over constants, for example HasDegree(Ana, PhD).

Examples:
Input:
1. All drones function correctly.
2. Every safe vehicle is thoroughly tested.
3. If a Python project is optimized, then it has clean code.
4. Sophia completed the core curriculum.
5. Are all drones functioning correctly?
Output:
["ForAll(x, FunctionCorrectly(x))",
"ForAll(x, Implies(Safe(x), ThoroughlyTested(x)))",
"ForAll(x, Implies(Optimized(x), CleanCode(x)))",
"CompletedCoreCurriculum(Sophia)",
"ForAll(x, FunctionCorrectly(x))"]
"""


def _normalize_options(options: Any) -> Dict[str, str]:
    if not options:
        return {}
    choices: Dict[str, str] = {}
    if isinstance(options, dict):
        for key, value in options.items():
            label = str(key).strip().upper()[:1]
            if label:
                choices[label] = str(value).strip()
        return choices
    if isinstance(options, list):
        for idx, item in enumerate(options):
            default_label = _LETTERS[idx] if idx < len(_LETTERS) else str(idx + 1)
            label = default_label
            text = item
            if isinstance(item, dict):
                label = str(item.get("label", item.get("key", default_label))).strip().upper()[:1]
                text = item.get("text", item.get("option", item.get("value", item.get("answer", ""))))
            else:
                match = _OPT.match(str(item))
                if match:
                    label, text = match.group(1).upper(), match.group(2)
            if label and str(text).strip():
                choices[label] = str(text).strip()
    return choices


def _parse_choices(question: str, options: Any = None) -> Tuple[str, Dict[str, str]]:
    explicit = _normalize_options(options)
    lines = str(question or "").splitlines()
    stem_lines: List[str] = []
    parsed: Dict[str, str] = {}
    for line in lines:
        match = _OPT.match(line)
        if match:
            parsed[match.group(1).upper()] = match.group(2).strip()
        else:
            stem_lines.append(line)
    return " ".join(x.strip() for x in stem_lines if x.strip()).strip(), (explicit or parsed)


def _safe_json_list(raw: str, expected_len: int) -> List[Optional[str]]:
    text = str(raw or "").strip()
    candidates = [text]
    first = text.find("[")
    last = text.rfind("]")
    if first >= 0 and last > first:
        candidates.insert(0, text[first:last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, list):
            out = [str(x).strip() if x is not None else None for x in parsed]
            if len(out) < expected_len:
                out += [None] * (expected_len - len(out))
            return out[:expected_len]
    return [None] * expected_len


def _convert_lines(lines: List[str], chat_fn: ChatFn, model_name: str, max_tokens: int, timeout: float) -> List[Optional[str]]:
    numbered = "\n".join(f"{idx + 1}. {line}" for idx, line in enumerate(lines))
    messages = [
        {"role": "system", "content": TYPE1_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Convert these lines using one shared predicate vocabulary.\n"
                f"Input:\n{numbered}\nOutput:"
            ),
        },
    ]
    raw = chat_fn(messages, model_name, max_tokens, timeout)
    return _safe_json_list(raw, expected_len=len(lines))


def _split_top_level(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _call_name_args(expr: str) -> Tuple[str, List[str]]:
    text = str(expr or "").strip()
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", text, flags=re.S)
    if not match:
        return text, []
    return match.group(1), _split_top_level(match.group(2))


class FOLParser:
    def __init__(self) -> None:
        self.obj = z3.DeclareSort("Obj")
        self.predicates: Dict[Tuple[str, int], Any] = {}
        self.props: Dict[str, Any] = {}
        self.scope: Dict[str, Any] = {}

    def _predicate(self, name: str, arity: int) -> Any:
        key = (name, arity)
        if key not in self.predicates:
            self.predicates[key] = z3.Function(name, *([self.obj] * arity), z3.BoolSort())
        return self.predicates[key]

    def _prop(self, name: str) -> Any:
        if name not in self.props:
            self.props[name] = z3.Bool(name)
        return self.props[name]

    def _arg(self, name: str) -> Any:
        clean = re.sub(r"\W+", "_", str(name).strip()) or "Unknown"
        return self.scope.get(clean, z3.Const(clean, self.obj))

    def parse(self, expr: Optional[str]) -> Optional[Any]:
        if expr is None:
            return None
        text = str(expr).strip().strip("`")
        if not text or re.search(r"\b(True|False|BoolVal)\b", text):
            return None
        return self._parse_expr(text)

    def _parse_expr(self, expr: str) -> Any:
        name, args = _call_name_args(expr)
        if not args:
            return self._prop(name)

        if name in {"ForAll", "Exists"} and len(args) == 2:
            var_name = re.sub(r"\W+", "_", args[0].strip()) or "x"
            old = self.scope.get(var_name)
            var = z3.Const(var_name, self.obj)
            self.scope[var_name] = var
            body = self._parse_expr(args[1])
            if old is None:
                self.scope.pop(var_name, None)
            else:
                self.scope[var_name] = old
            return z3.ForAll([var], body) if name == "ForAll" else z3.Exists([var], body)

        if name == "Not" and len(args) == 1:
            return z3.Not(self._parse_expr(args[0]))
        if name == "Implies" and len(args) == 2:
            return z3.Implies(self._parse_expr(args[0]), self._parse_expr(args[1]))
        if name == "And" and len(args) >= 2:
            return z3.And(*[self._parse_expr(arg) for arg in args])
        if name == "Or" and len(args) >= 2:
            return z3.Or(*[self._parse_expr(arg) for arg in args])

        zargs = [self._arg(arg) for arg in args]
        return self._predicate(name, len(zargs))(*zargs)


def _parse_many(parser: FOLParser, fol_lines: List[Optional[str]]) -> List[Optional[Any]]:
    parsed: List[Optional[Any]] = []
    for fol in fol_lines:
        try:
            parsed.append(parser.parse(fol))
        except Exception:
            parsed.append(None)
    return parsed


def _entails(premises: List[Any], query: Any, timeout_ms: int) -> Tuple[bool, List[int]]:
    if query is None:
        return False, []
    solver = z3.Solver()
    solver.set("timeout", int(timeout_ms))
    solver.set(unsat_core=True)
    trackers: List[Tuple[Any, int]] = []
    for idx, premise in enumerate(premises):
        if premise is None:
            continue
        tracker = z3.Bool(f"p{idx}")
        solver.assert_and_track(premise, tracker)
        trackers.append((tracker, idx))
    solver.add(z3.Not(query))
    if solver.check() == z3.unsat:
        core = {str(x) for x in solver.unsat_core()}
        return True, sorted(idx for tracker, idx in trackers if str(tracker) in core)
    return False, []


def _reason_yes_no(premises: List[Any], query: Any, timeout_ms: int) -> Tuple[str, List[int]]:
    ok, core = _entails(premises, query, timeout_ms)
    if ok:
        return "Yes", core
    ok, core = _entails(premises, z3.Not(query) if query is not None else None, timeout_ms)
    if ok:
        return "No", core
    return "Unknown", []


def _reason_mcq(premises: List[Any], options: Dict[str, Any], timeout_ms: int) -> Tuple[str, List[int]]:
    winners: List[Tuple[str, List[int]]] = []
    for label, query in options.items():
        ok, core = _entails(premises, query, timeout_ms)
        if ok:
            winners.append((label, core))
    if not winners:
        return "Unknown", []
    return min(winners, key=lambda item: len(item[1]) or 99)


def _explanation(answer: str, premises_nl: List[str], used: List[int], question: str) -> str:
    if answer == "Unknown":
        return (
            "The provided premises do not logically force either the statement or its negation. "
            "Therefore the safest answer is Unknown."
        )
    if not used:
        return f"The answer is {answer} because the converted premises entail the selected conclusion."
    cited = "; ".join(f"Premise {idx}: {premises_nl[idx - 1]}" for idx in used if 1 <= idx <= len(premises_nl))
    return f"The answer is {answer}. It follows from {cited}."


def run_type1_logic(
    payload: Dict[str, Any],
    chat_fn: ChatFn,
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    max_tokens: int = 1024,
    timeout_seconds: float = 25.0,
    z3_timeout_ms: int = 5000,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    started = time.perf_counter()
    query_id = str(payload.get("query_id") or payload.get("id") or "").strip()
    premises = payload.get("premises") or payload.get("premises-NL") or payload.get("premises_nl") or []
    if not isinstance(premises, list):
        premises = []
    premises_nl = [str(item).strip() for item in premises if str(item).strip()]
    question = str(payload.get("query") or payload.get("question") or "").strip()
    stem, choices = _parse_choices(question, payload.get("options"))

    if not premises_nl:
        raise ValueError("Type 1 logic route requires non-empty premises.")
    if not question:
        raise ValueError("Type 1 logic route requires a query/question.")

    qtype = "mcq" if len(choices) >= 2 else "yesno"
    target_texts = list(choices.values()) if qtype == "mcq" else [stem or question]
    all_lines = premises_nl + target_texts
    fol_lines = _convert_lines(all_lines, chat_fn, model_name, max_tokens=max_tokens, timeout=timeout_seconds)
    premise_fol = fol_lines[: len(premises_nl)]
    target_fol = fol_lines[len(premises_nl):]

    parser = FOLParser()
    premise_exprs = _parse_many(parser, premise_fol)
    target_exprs = _parse_many(parser, target_fol)

    if qtype == "mcq":
        option_exprs = {label: expr for label, expr in zip(choices.keys(), target_exprs)}
        answer, core0 = _reason_mcq(premise_exprs, option_exprs, timeout_ms=z3_timeout_ms)
    else:
        answer, core0 = _reason_yes_no(premise_exprs, target_exprs[0] if target_exprs else None, timeout_ms=z3_timeout_ms)

    used = [idx + 1 for idx in core0]
    result: Dict[str, Any] = {
        "query_id": query_id,
        "answer": answer,
        "unit": "",
        "explanation": _explanation(answer, premises_nl, used, stem or question),
        "premises_used": used,
        "reasoning": {
            "type": "evidence",
            "steps": [
                "Convert the natural-language premises and target statement into a shared first-order logic vocabulary.",
                "Use Z3 to test whether the premises entail the candidate conclusion.",
                f"Final answer: {answer}.",
            ],
        },
    }
    if debug:
        result["debug"] = {
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "method": "type1_logic_z3",
            "question_type": qtype,
            "model": model_name,
            "premise_fol": premise_fol,
            "target_fol": dict(zip(choices.keys(), target_fol)) if qtype == "mcq" else target_fol[:1],
            "confidence": 0.9 if answer != "Unknown" else 0.55,
        }
    return [result]
