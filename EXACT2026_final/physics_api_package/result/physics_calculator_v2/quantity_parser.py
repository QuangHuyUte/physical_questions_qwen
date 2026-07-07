from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class UnitInfo:
    canonical: str
    dimension: str
    scale_to_si: float


UNIT_TABLE: Dict[str, UnitInfo] = {
    "c": UnitInfo("C", "charge", 1.0),
    "coulomb": UnitInfo("C", "charge", 1.0),
    "coulombs": UnitInfo("C", "charge", 1.0),
    "mc": UnitInfo("mC", "charge", 1e-3),
    "millicoulomb": UnitInfo("mC", "charge", 1e-3),
    "millicoulombs": UnitInfo("mC", "charge", 1e-3),
    "uc": UnitInfo("uC", "charge", 1e-6),
    "microcoulomb": UnitInfo("uC", "charge", 1e-6),
    "microcoulombs": UnitInfo("uC", "charge", 1e-6),
    "nc": UnitInfo("nC", "charge", 1e-9),
    "nanocoulomb": UnitInfo("nC", "charge", 1e-9),
    "nanocoulombs": UnitInfo("nC", "charge", 1e-9),
    "a": UnitInfo("A", "current", 1.0),
    "ampere": UnitInfo("A", "current", 1.0),
    "amperes": UnitInfo("A", "current", 1.0),
    "amp": UnitInfo("A", "current", 1.0),
    "amps": UnitInfo("A", "current", 1.0),
    "ma": UnitInfo("mA", "current", 1e-3),
    "ua": UnitInfo("uA", "current", 1e-6),
    "v": UnitInfo("V", "voltage", 1.0),
    "volt": UnitInfo("V", "voltage", 1.0),
    "volts": UnitInfo("V", "voltage", 1.0),
    "mv": UnitInfo("mV", "voltage", 1e-3),
    "kv": UnitInfo("kV", "voltage", 1e3),
    "w": UnitInfo("W", "power", 1.0),
    "watt": UnitInfo("W", "power", 1.0),
    "watts": UnitInfo("W", "power", 1.0),
    "mw": UnitInfo("mW", "power", 1e-3),
    "kw": UnitInfo("kW", "power", 1e3),
    "ohm": UnitInfo("ohm", "resistance", 1.0),
    "ohms": UnitInfo("ohm", "resistance", 1.0),
    "omega": UnitInfo("ohm", "resistance", 1.0),
    "f": UnitInfo("F", "capacitance", 1.0),
    "farad": UnitInfo("F", "capacitance", 1.0),
    "farads": UnitInfo("F", "capacitance", 1.0),
    "uf": UnitInfo("uF", "capacitance", 1e-6),
    "microfarad": UnitInfo("uF", "capacitance", 1e-6),
    "microfarads": UnitInfo("uF", "capacitance", 1e-6),
    "nf": UnitInfo("nF", "capacitance", 1e-9),
    "pf": UnitInfo("pF", "capacitance", 1e-12),
    "h": UnitInfo("H", "inductance", 1.0),
    "henry": UnitInfo("H", "inductance", 1.0),
    "henries": UnitInfo("H", "inductance", 1.0),
    "mh": UnitInfo("mH", "inductance", 1e-3),
    "uh": UnitInfo("uH", "inductance", 1e-6),
    "j": UnitInfo("J", "energy", 1.0),
    "joule": UnitInfo("J", "energy", 1.0),
    "joules": UnitInfo("J", "energy", 1.0),
    "mj": UnitInfo("mJ", "energy", 1e-3),
    "uj": UnitInfo("uJ", "energy", 1e-6),
    "nj": UnitInfo("nJ", "energy", 1e-9),
    "n": UnitInfo("N", "force", 1.0),
    "newton": UnitInfo("N", "force", 1.0),
    "newtons": UnitInfo("N", "force", 1.0),
    "mn": UnitInfo("mN", "force", 1e-3),
    "m": UnitInfo("m", "length", 1.0),
    "meter": UnitInfo("m", "length", 1.0),
    "meters": UnitInfo("m", "length", 1.0),
    "metre": UnitInfo("m", "length", 1.0),
    "metres": UnitInfo("m", "length", 1.0),
    "cm": UnitInfo("cm", "length", 1e-2),
    "centimeter": UnitInfo("cm", "length", 1e-2),
    "centimeters": UnitInfo("cm", "length", 1e-2),
    "mm": UnitInfo("mm", "length", 1e-3),
    "m^2": UnitInfo("m^2", "area", 1.0),
    "m2": UnitInfo("m^2", "area", 1.0),
    "square meter": UnitInfo("m^2", "area", 1.0),
    "square meters": UnitInfo("m^2", "area", 1.0),
    "cm^2": UnitInfo("cm^2", "area", 1e-4),
    "cm2": UnitInfo("cm^2", "area", 1e-4),
    "square centimeter": UnitInfo("cm^2", "area", 1e-4),
    "square centimeters": UnitInfo("cm^2", "area", 1e-4),
    "mm^2": UnitInfo("mm^2", "area", 1e-6),
    "mm2": UnitInfo("mm^2", "area", 1e-6),
    "t": UnitInfo("T", "magnetic_field", 1.0),
    "tesla": UnitInfo("T", "magnetic_field", 1.0),
    "mt": UnitInfo("mT", "magnetic_field", 1e-3),
    "ut": UnitInfo("uT", "magnetic_field", 1e-6),
    "wb": UnitInfo("Wb", "magnetic_flux", 1.0),
    "weber": UnitInfo("Wb", "magnetic_flux", 1.0),
    "webers": UnitInfo("Wb", "magnetic_flux", 1.0),
    "hz": UnitInfo("Hz", "frequency", 1.0),
    "s": UnitInfo("s", "time", 1.0),
    "second": UnitInfo("s", "time", 1.0),
    "seconds": UnitInfo("s", "time", 1.0),
    "ms": UnitInfo("ms", "time", 1e-3),
    "us": UnitInfo("us", "time", 1e-6),
    "v/m": UnitInfo("V/m", "electric_field", 1.0),
    "n/c": UnitInfo("N/C", "electric_field", 1.0),
    "rad/s": UnitInfo("rad/s", "angular_frequency", 1.0),
    "degree": UnitInfo("degree", "angle", math.pi / 180.0),
    "degrees": UnitInfo("degree", "angle", math.pi / 180.0),
    "deg": UnitInfo("degree", "angle", math.pi / 180.0),
    "rad": UnitInfo("rad", "angle", 1.0),
    "%": UnitInfo("%", "percent", 1.0),
    "turns/m": UnitInfo("turns/m", "turn_density", 1.0),
    "turn": UnitInfo("turns", "turns", 1.0),
    "turns": UnitInfo("turns", "turns", 1.0),
    "times": UnitInfo("times", "factor", 1.0),
    "ohm m": UnitInfo("ohm m", "resistivity", 1.0),
    "ohm*m": UnitInfo("ohm m", "resistivity", 1.0),
    "ohmm": UnitInfo("ohm m", "resistivity", 1.0),
    "ohm.m": UnitInfo("ohm m", "resistivity", 1.0),
    "ohm meter": UnitInfo("ohm m", "resistivity", 1.0),
    "ohm meters": UnitInfo("ohm m", "resistivity", 1.0),
    "ohm metre": UnitInfo("ohm m", "resistivity", 1.0),
    "ohm metres": UnitInfo("ohm m", "resistivity", 1.0),
}


ROLE_ALIASES = {
    "potential_difference": "voltage",
    "emf": "voltage",
    "voltage_error": "voltage",
    "current_error": "current",
    "resistance_error": "resistance_uncertainty",
    "delta_r": "resistance_uncertainty",
    "absolute_error": "absolute_error",
    "relative_error": "relative_error",
    "electric_force": "force",
    "coulomb_force": "force",
    "field": "electric_field",
    "electric_field_strength": "electric_field",
    "magnetic_flux_density": "magnetic_field",
    "flux": "magnetic_flux",
    "separation": "distance",
    "radius": "distance",
    "length": "distance",
    "plate_area": "area",
    "surface_area": "area",
    "maximum_current": "current",
    "max_current": "current",
    "maximum_voltage": "voltage",
    "max_voltage": "voltage",
    "stored_energy": "energy",
    "magnetic_energy": "energy",
    "electric_energy": "energy",
    "inductive_reactance": "inductive_reactance",
    "capacitive_reactance": "capacitive_reactance",
    "reactance": "reactance",
    "impedance": "impedance",
    "angular_speed": "angular_frequency",
    "time_interval": "time",
    "frequency_factor": "frequency_factor",
    "factor": "frequency_factor",
    "scale_factor": "frequency_factor",
    "turn_count": "turns",
    "number_of_turns": "turns",
    "rho": "resistivity",
    "resistivity": "resistivity",
    "wire_resistivity": "resistivity",
    "cross_section": "area",
    "cross_sectional_area": "area",
    "section_area": "area",
    "dielectric": "dielectric_constant",
    "dielectric_constant": "dielectric_constant",
    "relative_permittivity": "dielectric_constant",
    "epsilon_r": "dielectric_constant",
    "least_count": "least_count",
    "instrument_resolution": "least_count",
    "measured": "measured_value",
    "measured_value": "measured_value",
    "actual": "actual_value",
    "actual_value": "actual_value",
}


def normalize_text(text: Any) -> str:
    value = str(text or "")
    value = value.replace("\u03bc", "u").replace("\u00b5", "u")
    value = value.replace("\u03a9", "ohm").replace("\u03c9", "omega")
    value = value.replace("\u00b2", "^2").replace("\u00b3", "^3")
    value = value.replace("\u2212", "-").replace("\u00d7", "x")
    return re.sub(r"\s+", " ", value.strip())


def normalize_unit(unit_text: Any) -> str:
    unit = normalize_text(unit_text).lower().strip()
    unit = unit.replace("micro ", "micro")
    unit = unit.replace("square centimetres", "square centimeters")
    unit = unit.replace("square metres", "square meters")
    unit = unit.replace("ohm*m", "ohm m")
    unit = unit.replace(" ", " ")
    return unit


def unit_info(unit_text: Any) -> UnitInfo:
    unit = normalize_unit(unit_text)
    if unit in UNIT_TABLE:
        return UNIT_TABLE[unit]
    compact = unit.replace(" ", "")
    if compact in UNIT_TABLE:
        return UNIT_TABLE[compact]
    return UnitInfo(str(unit_text or ""), "unknown", 1.0)


def parse_value_text(value_text: Any) -> Optional[float]:
    if isinstance(value_text, (int, float)):
        return float(value_text)
    text = normalize_text(value_text)
    if not text:
        return None
    text = text.strip()
    sci = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*(?:x|\*)\s*10\^?([+-]?\d+)$", text, re.I)
    if sci:
        return float(sci.group(1)) * (10.0 ** int(sci.group(2)))
    frac = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)$", text)
    if frac and float(frac.group(2)) != 0:
        return float(frac.group(1)) / float(frac.group(2))
    match = re.search(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, re.I)
    if not match:
        return None
    return float(match.group(0))


def role_key(role: Any, name: Any = "") -> str:
    raw = normalize_text(role or name).lower()
    raw = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    return ROLE_ALIASES.get(raw, raw)


def raw_span_present(question: str, raw_span: Any) -> bool:
    span = normalize_text(raw_span).lower()
    if not span:
        return False
    q = normalize_text(question).lower()
    if span in q:
        return True
    compact_q = re.sub(r"[^a-z0-9.+-]+", "", q)
    compact_span = re.sub(r"[^a-z0-9.+-]+", "", span)
    return bool(compact_span and compact_span in compact_q)


def _parse_uncertainty_from_qualifiers(qualifiers: Iterable[Any], fallback_unit: Any) -> Optional[float]:
    for qualifier in qualifiers or []:
        text = normalize_text(qualifier).lower()
        if not any(word in text for word in ("uncertainty", "error", "delta", "+/-", "+-")):
            continue
        value = parse_value_text(text)
        if value is None:
            continue
        unit_match = re.search(
            r"(microcoulombs?|microfarads?|microhenr(?:y|ies)|microamps?|microamperes?|"
            r"millicoulombs?|millifarads?|millihenr(?:y|ies)|milliamps?|milliamperes?|"
            r"volts?|amperes?|amps?|ohms?|watts?|joules?|newtons?|meters?|metres?|centimeters?|seconds?|"
            r"u[cfahtj]|m[cfahtjv]|[cvafhjwn])\b",
            text,
        )
        info = unit_info(unit_match.group(0) if unit_match else fallback_unit)
        return abs(value) * info.scale_to_si
    return None


def parse_given(given: Dict[str, Any], question: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(given, dict):
        return None
    role = role_key(given.get("role"), given.get("name") or given.get("symbol"))
    unit_text = given.get("unit_text", given.get("unit", given.get("si_unit", "")))
    info = unit_info(unit_text)
    raw_value = given.get("value_text", given.get("value", None))
    if raw_value is None and "si_value" in given:
        raw_value = given.get("si_value")
        info = UnitInfo(info.canonical or str(unit_text), info.dimension, 1.0)
    value = parse_value_text(raw_value)
    if value is None:
        return None
    sign = str(given.get("sign") or "").lower()
    if sign == "negative" and value > 0:
        value = -value
    si_value = value * info.scale_to_si
    qualifiers = given.get("qualifiers") or []
    uncertainty = _parse_uncertainty_from_qualifiers(qualifiers, unit_text)
    raw_span = given.get("raw_span") or given.get("source") or ""
    return {
        "name": str(given.get("name") or given.get("symbol") or role),
        "role": role,
        "dimension": info.dimension if info.dimension != "unknown" else role,
        "raw_span": raw_span,
        "raw_span_ok": raw_span_present(question, raw_span) if question else True,
        "value": value,
        "si_value": si_value,
        "unit": info.canonical or str(unit_text or ""),
        "unit_text": str(unit_text or ""),
        "uncertainty_si": uncertainty,
        "qualifiers": list(qualifiers) if isinstance(qualifiers, list) else [str(qualifiers)],
    }


def target_unit(payload: Dict[str, Any]) -> str:
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    unit = target.get("unit") or target.get("unit_text") or ""
    info = unit_info(unit)
    return info.canonical or str(unit or "")


def target_role(payload: Dict[str, Any]) -> str:
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    return role_key(target.get("role"), target.get("symbol"))
