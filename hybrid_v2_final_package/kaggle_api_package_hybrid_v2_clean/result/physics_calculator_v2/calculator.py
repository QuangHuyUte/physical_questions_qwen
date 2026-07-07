from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional

from .formula_bank import describe_formula, ordered_formula_ids
from .quantity_parser import parse_given, target_role, target_unit, unit_info


K = 9.0e9
MU0 = 4.0 * math.pi * 1.0e-7
EPS0 = 8.854187817e-12


class CalculationError(Exception):
    pass


def _format_number(value: float) -> str:
    if abs(value) < 1e-15:
        return "0"
    if abs(value - round(value)) < 1e-10 and abs(value) < 1e9:
        return str(int(round(value)))
    return f"{value:.6g}"


def _convert_from_si(value_si: float, out_unit: str) -> float:
    info = unit_info(out_unit)
    scale = info.scale_to_si or 1.0
    return value_si / scale


def _givens(payload: Dict[str, Any], question: str) -> list[Dict[str, Any]]:
    values = []
    for given in payload.get("givens") or []:
        parsed = parse_given(given, question=question)
        if parsed is not None:
            values.append(parsed)
    return values


def _by_role(givens: list[Dict[str, Any]], *roles: str) -> list[Dict[str, Any]]:
    wanted = set(roles)
    return [g for g in givens if g["role"] in wanted or g["dimension"] in wanted]


def _first(givens: list[Dict[str, Any]], *roles: str) -> Optional[float]:
    values = _by_role(givens, *roles)
    return values[0]["si_value"] if values else None


def _values(givens: list[Dict[str, Any]], *roles: str) -> list[float]:
    return [g["si_value"] for g in _by_role(givens, *roles)]


def _delta_from_values(values: list[float]) -> Optional[float]:
    if len(values) >= 2:
        return abs(values[0] - values[1])
    if len(values) == 1:
        return abs(values[0])
    return None


def _text(payload: Dict[str, Any]) -> str:
    parts = [str(payload.get("topic") or "")]
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    parts.extend(str(v) for v in target.values())
    parts.extend(str(v) for v in payload.get("constraints") or [])
    for item in payload.get("formula_candidates") or []:
        parts.append(str(item))
    return " ".join(parts).lower()


def _target_matches(payload: Dict[str, Any], *roles: str) -> bool:
    role = target_role(payload)
    unit_role = unit_info(target_unit(payload)).dimension
    return role in roles or unit_role in roles


def _needs_series(payload: Dict[str, Any]) -> bool:
    return "series" in _text(payload)


def _needs_parallel(payload: Dict[str, Any]) -> bool:
    return "parallel" in _text(payload)


def _angle_cos(givens: list[Dict[str, Any]], payload: Dict[str, Any]) -> float:
    angles = _values(givens, "angle")
    if angles:
        return math.cos(angles[0])
    text = _text(payload)
    if "normal" in text and ("aligned" in text or "parallel" in text):
        return 1.0
    if "perpendicular" in text:
        return 0.0
    if "right angle" in text or "90" in text:
        return 0.0
    if "equilateral" in text or "60" in text:
        return 0.5
    if "opposite" in text or "180" in text:
        return -1.0
    return 1.0


def _measurement_pairs(givens: list[Dict[str, Any]]) -> list[tuple[float, float, str]]:
    pairs = []
    for given in givens:
        if given.get("uncertainty_si") is not None:
            pairs.append((abs(float(given["si_value"])), abs(float(given["uncertainty_si"])), given["role"]))
    return pairs


def _result(
    payload: Dict[str, Any],
    givens: list[Dict[str, Any]],
    formula_id: str,
    value_si: float,
    out_unit: str,
    calc: str,
) -> Dict[str, Any]:
    value_out = _convert_from_si(value_si, out_unit)
    answer = _format_number(value_out)
    base_conf = float(payload.get("confidence") or 0.7)
    raw_span_penalty = 0.08 if any(not g.get("raw_span_ok") for g in givens) else 0.0
    confidence = max(0.0, min(0.92, base_conf - raw_span_penalty))
    topic = str(payload.get("topic") or "")
    return {
        "answer": answer,
        "unit": out_unit,
        "confidence": confidence,
        "method": "calculator_v2_numeric_parser",
        "topic": topic,
        "formula_id": formula_id,
        "formula": describe_formula(formula_id),
        "calculation": calc,
        "trace": {
            "topic": topic,
            "target": payload.get("target") or {},
            "formula_id": formula_id,
            "formula": describe_formula(formula_id),
            "givens": [
                {
                    "name": g["name"],
                    "role": g["role"],
                    "value_si": g["si_value"],
                    "unit": g["unit"],
                    "raw_span": g["raw_span"],
                    "raw_span_ok": g["raw_span_ok"],
                }
                for g in givens
            ],
            "calculation": calc,
            "final_answer": {"answer": answer, "unit": out_unit},
            "guardrails": [
                "The parser is not allowed to provide an answer or executable code.",
                "The calculator recomputes the final value from extracted quantities.",
                "Raw spans are checked against the original question when available.",
            ],
        },
    }


def _require(value: Optional[float], name: str) -> float:
    if value is None:
        raise CalculationError(f"missing_{name}")
    return value


def _solve_formula(formula_id: str, payload: Dict[str, Any], givens: list[Dict[str, Any]]) -> tuple[float, str]:
    q = _values(givens, "charge")
    r = _first(givens, "distance", "length")
    U = _first(givens, "voltage")
    I = _first(givens, "current")
    P = _first(givens, "power")
    R_values = _values(givens, "resistance")
    R = R_values[0] if R_values else None
    Z = _first(givens, "impedance")
    C_values = _values(givens, "capacitance")
    C = C_values[0] if C_values else None
    L = _first(givens, "inductance")
    W = _first(givens, "energy")
    f = _first(givens, "frequency")
    frequency_factor = _first(givens, "frequency_factor")
    F = _first(givens, "force")
    E = _first(givens, "electric_field")
    B = _first(givens, "magnetic_field")
    area = _first(givens, "area")
    phi = _first(givens, "magnetic_flux")
    time_value = _first(givens, "time")
    n_turns = _first(givens, "turn_density")
    turns = _first(givens, "turns")
    XL = _first(givens, "inductive_reactance")
    XC = _first(givens, "capacitive_reactance")
    reactances = _values(givens, "reactance")
    if XL is None and len(reactances) >= 1:
        XL = reactances[0]
    if XC is None and len(reactances) >= 2:
        XC = reactances[1]

    if formula_id == "resistance_from_voltage_current":
        return _require(U, "U") / _require(I, "I"), "R = U/I"
    if formula_id == "resistance_from_resonance_impedance":
        value = Z if Z is not None else R
        return _require(value, "Z"), "At resonance, R = Z"
    if formula_id == "current_from_voltage_resistance":
        return _require(U, "U") / _require(R, "R"), "I = U/R"
    if formula_id == "voltage_from_current_resistance":
        return _require(I, "I") * _require(R, "R"), "U = I*R"
    if formula_id == "power_from_voltage_current":
        return _require(U, "U") * _require(I, "I"), "P = U*I"
    if formula_id == "power_from_current_resistance":
        return _require(I, "I") ** 2 * _require(R, "R"), "P = I^2*R"
    if formula_id == "power_from_voltage_resistance":
        return _require(U, "U") ** 2 / _require(R, "R"), "P = U^2/R"
    if formula_id == "power_from_energy_time":
        return _require(W, "A") / _require(time_value, "t"), "P = A/t"
    if formula_id == "power_total_sum":
        power_values = _values(givens, "power")
        if len(power_values) < 2:
            raise CalculationError("need_multiple_powers")
        return sum(power_values), "P_total = sum(P_i)"
    if formula_id == "current_from_power_voltage":
        return _require(P, "P") / _require(U, "U"), "I = P/U"
    if formula_id == "voltage_from_power_current":
        return _require(P, "P") / _require(I, "I"), "U = P/I"
    if formula_id == "current_from_power_resistance":
        return math.sqrt(_require(P, "P") / _require(R, "R")), "I = sqrt(P/R)"
    if formula_id == "voltage_from_power_resistance":
        return math.sqrt(_require(P, "P") * _require(R, "R")), "U = sqrt(P*R)"
    if formula_id == "wire_resistance_from_resistivity_length_area":
        rho = _first(givens, "resistivity")
        return _require(rho, "rho") * _require(r, "l") / _require(area, "S"), "R = rho*l/S"
    if formula_id == "resistance_series":
        if len(R_values) < 2:
            raise CalculationError("need_multiple_resistances")
        return sum(R_values), "R_eq = sum(R_i)"
    if formula_id == "resistance_parallel":
        if len(R_values) < 2:
            raise CalculationError("need_multiple_resistances")
        return 1.0 / sum(1.0 / value for value in R_values), "1/R_eq = sum(1/R_i)"
    if formula_id == "total_current_parallel_resistors_from_voltage":
        if len(R_values) < 2:
            raise CalculationError("need_multiple_resistances")
        return _require(U, "U") * sum(1.0 / value for value in R_values), "I_total = U*sum(1/R_i)"
    if formula_id == "capacitance_series":
        if len(C_values) < 2:
            raise CalculationError("need_multiple_capacitances")
        return 1.0 / sum(1.0 / value for value in C_values), "1/C_eq = sum(1/C_i)"
    if formula_id == "capacitance_parallel":
        if len(C_values) < 2:
            raise CalculationError("need_multiple_capacitances")
        return sum(C_values), "C_eq = sum(C_i)"
    if formula_id == "capacitor_charge_from_C_U":
        return _require(C, "C") * _require(U, "U"), "Q = C*U"
    if formula_id == "capacitor_capacitance_from_charge_voltage":
        charge = q[0] if q else None
        return abs(_require(charge, "Q")) / _require(U, "U"), "C = Q/U"
    if formula_id == "capacitor_voltage_from_charge_capacitance":
        charge = q[0] if q else None
        return abs(_require(charge, "Q")) / _require(C, "C"), "U = Q/C"
    if formula_id == "capacitor_energy_from_C_U":
        return 0.5 * _require(C, "C") * _require(U, "U") ** 2, "W = 0.5*C*U^2"
    if formula_id == "capacitor_energy_from_Q_U":
        charge = q[0] if q else None
        return 0.5 * abs(_require(charge, "Q")) * _require(U, "U"), "W = 0.5*Q*U"
    if formula_id == "capacitor_energy_from_Q_C":
        charge = q[0] if q else None
        return (_require(charge, "Q") ** 2) / (2.0 * _require(C, "C")), "W = Q^2/(2C)"
    if formula_id == "capacitor_capacitance_from_energy_voltage":
        return 2.0 * _require(W, "W") / (_require(U, "U") ** 2), "C = 2W/U^2"
    if formula_id == "capacitor_voltage_from_energy_capacitance":
        return math.sqrt(2.0 * _require(W, "W") / _require(C, "C")), "U = sqrt(2W/C)"
    if formula_id == "parallel_plate_capacitance_with_dielectric":
        eps_r = _first(givens, "dielectric_constant")
        return _require(eps_r, "eps_r") * EPS0 * _require(area, "A") / _require(r, "d"), "C = eps_r*eps0*A/d"
    if formula_id == "parallel_plate_field_from_voltage_distance":
        return _require(U, "U") / _require(r, "d"), "E = U/d"
    if formula_id == "inductor_energy_from_L_I":
        return 0.5 * _require(L, "L") * _require(I, "I") ** 2, "W = 0.5*L*I^2"
    if formula_id == "inductance_from_energy_current":
        return 2.0 * _require(W, "W") / (_require(I, "I") ** 2), "L = 2W/I^2"
    if formula_id == "inductor_current_from_energy_inductance":
        return math.sqrt(2.0 * _require(W, "W") / _require(L, "L")), "I = sqrt(2W/L)"
    if formula_id == "lc_max_current_from_voltage_capacitance_inductance":
        return _require(U, "U") * math.sqrt(_require(C, "C") / _require(L, "L")), "I0 = U0*sqrt(C/L)"
    if formula_id == "lc_angular_frequency_from_L_C":
        return 1.0 / math.sqrt(_require(L, "L") * _require(C, "C")), "omega = 1/sqrt(LC)"
    if formula_id == "lc_frequency_from_L_C":
        return 1.0 / (2.0 * math.pi * math.sqrt(_require(L, "L") * _require(C, "C"))), "f = 1/(2*pi*sqrt(LC))"
    if formula_id == "rlc_resonance_capacitance_from_L_f":
        return 1.0 / (((2.0 * math.pi * _require(f, "f")) ** 2) * _require(L, "L")), "C = 1/((2*pi*f)^2*L)"
    if formula_id == "rlc_resonance_inductance_from_C_f":
        return 1.0 / (((2.0 * math.pi * _require(f, "f")) ** 2) * _require(C, "C")), "L = 1/((2*pi*f)^2*C)"
    if formula_id == "parallel_plate_capacitance_from_area_distance":
        return EPS0 * _require(area, "A") / _require(r, "d"), "C = eps0*A/d"
    if formula_id == "coulomb_force_two_charges":
        if len(q) < 2:
            raise CalculationError("need_two_charges")
        return K * abs(q[0] * q[1]) / (_require(r, "r") ** 2), "F = k*|q1*q2|/r^2"
    if formula_id == "coulomb_force_right_angle_equal_charges":
        charge = q[0] if q else None
        single = K * abs(_require(charge, "q")) ** 2 / (_require(r, "r") ** 2)
        return math.sqrt(2.0) * single, "F_net = sqrt(2)*k*q^2/r^2"
    if formula_id == "coulomb_force_equilateral_equal_charges":
        charge = q[0] if q else None
        single = K * abs(_require(charge, "q")) ** 2 / (_require(r, "r") ** 2)
        return math.sqrt(3.0) * single, "F_net = sqrt(3)*k*q^2/r^2"
    if formula_id == "force_from_charge_field":
        charge = q[0] if q else None
        return abs(_require(charge, "q")) * _require(E, "E"), "F = |q|E"
    if formula_id == "electric_field_from_force_charge":
        charge = q[0] if q else None
        return _require(F, "F") / abs(_require(charge, "q")), "E = F/|q|"
    if formula_id == "electric_field_point_charge_or_superposition":
        charge = q[0] if q else None
        return K * abs(_require(charge, "q")) / (_require(r, "r") ** 2), "E = k*|q|/r^2"
    if formula_id == "electric_field_two_charges_angle":
        if len(q) < 2:
            raise CalculationError("need_two_charges")
        distances = _values(givens, "distance", "length")
        r1 = distances[0] if distances else None
        r2 = distances[1] if len(distances) >= 2 else r1
        e1 = K * abs(q[0]) / (_require(r1, "r1") ** 2)
        e2 = K * abs(q[1]) / (_require(r2, "r2") ** 2)
        cos_theta = _angle_cos(givens, payload)
        return math.sqrt(max(0.0, e1**2 + e2**2 + 2.0 * e1 * e2 * cos_theta)), "E = sqrt(E1^2+E2^2+2E1E2*cos(theta))"
    if formula_id == "resultant_two_vectors_from_angle":
        e_values = _values(givens, "electric_field")
        f_values = _values(givens, "force")
        values = e_values if _target_matches(payload, "electric_field") else f_values
        if len(values) < 2:
            raise CalculationError("need_two_vectors")
        a, b = values[0], values[1]
        cos_theta = _angle_cos(givens, payload)
        return math.sqrt(max(0.0, a**2 + b**2 + 2.0 * a * b * cos_theta)), "R = sqrt(A^2+B^2+2AB*cos(theta))"
    if formula_id == "electric_field_resultant_two_vectors_from_angle":
        values = _values(givens, "electric_field")
        if len(values) < 2:
            raise CalculationError("need_two_electric_fields")
        e1, e2 = values[0], values[1]
        cos_theta = _angle_cos(givens, payload)
        return math.sqrt(max(0.0, e1**2 + e2**2 + 2.0 * e1 * e2 * cos_theta)), "E = sqrt(E1^2+E2^2+2E1E2*cos(theta))"
    if formula_id == "magnetic_flux_from_B_area_angle":
        return _require(B, "B") * _require(area, "S") * _angle_cos(givens, payload), "Phi = B*S*cos(theta)"
    if formula_id == "magnetic_flux_linkage_from_turns_flux":
        return _require(turns, "N") * _require(phi, "Phi"), "lambda = N*Phi"
    if formula_id == "magnetic_flux_from_solenoid_area":
        return MU0 * _require(n_turns, "n") * _require(I, "I") * _require(area, "S"), "Phi = mu0*n*I*S"
    if formula_id == "faraday_emf_from_flux_change":
        phi_values = _values(givens, "magnetic_flux")
        if len(phi_values) >= 2:
            delta_phi = abs(phi_values[0] - phi_values[1])
        else:
            delta_phi = abs(_require(phi, "DeltaPhi"))
        return delta_phi / _require(time_value, "DeltaT"), "|e| = |Delta Phi|/Delta t"
    if formula_id == "self_induction_emf_from_L_current_change":
        delta_i = _delta_from_values(_values(givens, "current"))
        return _require(L, "L") * _require(delta_i, "DeltaI") / _require(time_value, "DeltaT"), "|e| = L*|Delta I|/Delta t"
    if formula_id == "inductance_from_emf_current_change":
        delta_i = _delta_from_values(_values(givens, "current"))
        return _require(U, "e") * _require(time_value, "DeltaT") / _require(delta_i, "DeltaI"), "L = |e|*Delta t/|Delta I|"
    if formula_id == "solenoid_field_from_turn_density_current":
        return MU0 * _require(n_turns, "n") * _require(I, "I"), "B = mu0*n*I"
    if formula_id == "solenoid_field_from_turns_length_current":
        return MU0 * (_require(turns, "N") / _require(r, "l")) * _require(I, "I"), "B = mu0*(N/l)*I"
    if formula_id == "series_ac_impedance_from_R_XL_XC":
        return math.sqrt(_require(R, "R") ** 2 + (_require(XL, "XL") - _require(XC, "XC")) ** 2), "Z = sqrt(R^2+(XL-XC)^2)"
    if formula_id == "inductive_reactance_from_L_f":
        return 2.0 * math.pi * _require(f, "f") * _require(L, "L"), "X_L = 2*pi*f*L"
    if formula_id == "capacitive_reactance_from_C_f":
        return 1.0 / (2.0 * math.pi * _require(f, "f") * _require(C, "C")), "X_C = 1/(2*pi*f*C)"
    if formula_id == "rlc_series_current_from_U_R_XL_XC":
        z_value = math.sqrt(_require(R, "R") ** 2 + (_require(XL, "XL") - _require(XC, "XC")) ** 2)
        return _require(U, "U") / z_value, "I = U/Z"
    if formula_id == "rlc_series_power_from_U_R_XL_XC":
        z2 = _require(R, "R") ** 2 + (_require(XL, "XL") - _require(XC, "XC")) ** 2
        return (_require(U, "U") ** 2) * _require(R, "R") / z2, "P = U^2*R/Z^2"
    if formula_id == "rlc_frequency_scaled_current":
        k = _require(frequency_factor, "frequency_factor")
        xl_new = _require(XL, "XL") * k
        xc_new = _require(XC, "XC") / k
        z_value = math.sqrt(_require(R, "R") ** 2 + (xl_new - xc_new) ** 2)
        return _require(U, "U") / z_value, "I = U/sqrt(R^2+(kXL-XC/k)^2)"
    if formula_id == "rlc_frequency_scaled_resistor_voltage":
        k = _require(frequency_factor, "frequency_factor")
        xl_new = _require(XL, "XL") * k
        xc_new = _require(XC, "XC") / k
        if abs(xl_new - xc_new) > 1e-6 * max(abs(xl_new), abs(xc_new), 1.0):
            raise CalculationError("not_resonant_after_frequency_scaling")
        return _require(U, "U"), "After scaling, X_L' = X_C', so U_R = U"
    if formula_id == "angular_frequency_from_frequency":
        return 2.0 * math.pi * _require(f, "f"), "omega = 2*pi*f"
    if formula_id == "relative_error_from_absolute":
        pairs = _measurement_pairs(givens)
        if pairs:
            value, delta, _role = pairs[0]
            return delta / value * 100.0, "relative error = Delta x/x*100%"
        abs_error = _first(givens, "absolute_error")
        measured = next((g["si_value"] for g in givens if g["role"] not in {"absolute_error", "relative_error"}), None)
        return _require(abs_error, "Delta") / _require(measured, "x") * 100.0, "relative error = Delta x/x*100%"
    if formula_id == "absolute_error_from_relative":
        relative = _first(givens, "relative_error")
        measured = _first(givens, "measured_value")
        if measured is None:
            measured = next((g["si_value"] for g in givens if g["role"] not in {"absolute_error", "relative_error"}), None)
        return _require(relative, "relative_error") * _require(measured, "x") / 100.0, "Delta x = relative_error*x/100"
    if formula_id == "absolute_error_from_least_count_half":
        least_count = _first(givens, "least_count")
        return _require(least_count, "least_count") / 2.0, "Delta x = least_count/2"
    if formula_id == "relative_error_product_or_quotient":
        pairs = _measurement_pairs(givens)
        if len(pairs) < 2:
            raise CalculationError("need_multiple_uncertainty_pairs")
        return sum(delta / value for value, delta, _role in pairs) * 100.0, "relative error = sum(Delta x/x)*100%"
    if formula_id == "resistance_uncertainty_from_voltage_current":
        pairs = {role: (value, delta) for value, delta, role in _measurement_pairs(givens)}
        if "voltage" not in pairs or "current" not in pairs:
            raise CalculationError("need_voltage_current_uncertainties")
        U_value, dU = pairs["voltage"]
        I_value, dI = pairs["current"]
        R_value = U_value / I_value
        return R_value * (dU / U_value + dI / I_value), "Delta R = R(Delta U/U + Delta I/I)"
    if formula_id == "absolute_error_difference":
        measured = _first(givens, "measured_value")
        actual = _first(givens, "actual_value")
        return abs(_require(measured, "measured") - _require(actual, "actual")), "Delta x = |x_measured - x_actual|"
    raise CalculationError(f"unsupported_formula:{formula_id}")


def _infer_formula_ids(payload: Dict[str, Any], givens: list[Dict[str, Any]]) -> list[str]:
    ids = ordered_formula_ids(payload)
    text = _text(payload)
    target = target_role(payload)
    has_explicit_candidate = bool(payload.get("formula_candidates"))
    ac_reactance_context = "reactance" in text or target == "impedance"
    if not has_explicit_candidate and _target_matches(payload, "resistance") and _needs_parallel(payload):
        ids.insert(0, "resistance_parallel")
    if not has_explicit_candidate and _target_matches(payload, "resistance") and _needs_series(payload) and not ac_reactance_context:
        ids.insert(0, "resistance_series")
    if _target_matches(payload, "resistance") and _by_role(givens, "resistivity") and _by_role(givens, "area"):
        ids.insert(0, "wire_resistance_from_resistivity_length_area")
    if _target_matches(payload, "current") and _by_role(givens, "power") and _by_role(givens, "resistance"):
        ids.insert(0, "current_from_power_resistance")
    if _target_matches(payload, "voltage") and _by_role(givens, "power") and _by_role(givens, "resistance"):
        ids.insert(0, "voltage_from_power_resistance")
    if _target_matches(payload, "capacitance") and _needs_parallel(payload):
        ids.insert(0, "capacitance_parallel")
    if _target_matches(payload, "capacitance") and _needs_series(payload):
        ids.insert(0, "capacitance_series")
    if _target_matches(payload, "capacitance") and _by_role(givens, "charge") and _by_role(givens, "voltage"):
        ids.insert(0, "capacitor_capacitance_from_charge_voltage")
    if _target_matches(payload, "voltage") and _by_role(givens, "charge") and _by_role(givens, "capacitance"):
        ids.insert(0, "capacitor_voltage_from_charge_capacitance")
    if _target_matches(payload, "energy") and _by_role(givens, "charge") and _by_role(givens, "voltage"):
        ids.insert(0, "capacitor_energy_from_Q_U")
    if _target_matches(payload, "energy") and _by_role(givens, "charge") and _by_role(givens, "capacitance"):
        ids.insert(0, "capacitor_energy_from_Q_C")
    if _target_matches(payload, "capacitance") and _by_role(givens, "dielectric_constant"):
        ids.insert(0, "parallel_plate_capacitance_with_dielectric")
    if _target_matches(payload, "electric_field") and _by_role(givens, "voltage") and _by_role(givens, "distance"):
        ids.insert(0, "parallel_plate_field_from_voltage_distance")
    if _target_matches(payload, "current") and _by_role(givens, "energy") and _by_role(givens, "inductance"):
        ids.insert(0, "inductor_current_from_energy_inductance")
    if "resonance" in text and _target_matches(payload, "capacitance"):
        ids.insert(0, "rlc_resonance_capacitance_from_L_f")
    if "resonance" in text and _target_matches(payload, "inductance"):
        ids.insert(0, "rlc_resonance_inductance_from_C_f")
    if _target_matches(payload, "force") and "right" in text and _by_role(givens, "charge") and _by_role(givens, "distance"):
        ids.insert(0, "coulomb_force_right_angle_equal_charges")
    if _target_matches(payload, "force") and "equilateral" in text and _by_role(givens, "charge") and _by_role(givens, "distance"):
        ids.insert(0, "coulomb_force_equilateral_equal_charges")
    if _target_matches(payload, "electric_field") and len(_by_role(givens, "charge")) >= 2 and _by_role(givens, "distance"):
        ids.insert(0, "electric_field_two_charges_angle")
    if _target_matches(payload, "electric_field") and len(_by_role(givens, "electric_field")) >= 2:
        ids.insert(0, "electric_field_resultant_two_vectors_from_angle")
    if _target_matches(payload, "force") and len(_by_role(givens, "force")) >= 2:
        ids.insert(0, "resultant_two_vectors_from_angle")
    if _target_matches(payload, "magnetic_flux"):
        ids.insert(0, "magnetic_flux_from_B_area_angle")
    if _target_matches(payload, "resistance") and _by_role(givens, "capacitance") and _by_role(givens, "frequency"):
        ids.insert(0, "capacitive_reactance_from_C_f")
    if _target_matches(payload, "current") and _by_role(givens, "voltage") and _by_role(givens, "inductive_reactance", "reactance"):
        ids.insert(0, "rlc_series_current_from_U_R_XL_XC")
    if _target_matches(payload, "power") and _by_role(givens, "voltage") and _by_role(givens, "inductive_reactance", "reactance"):
        ids.insert(0, "rlc_series_power_from_U_R_XL_XC")
    if _target_matches(payload, "relative_error"):
        ids.insert(0, "relative_error_from_absolute")
    if _target_matches(payload, "absolute_error") and _by_role(givens, "relative_error"):
        ids.insert(0, "absolute_error_from_relative")
    if _target_matches(payload, "absolute_error") and _by_role(givens, "least_count"):
        ids.insert(0, "absolute_error_from_least_count_half")
    if _target_matches(payload, "resistance_uncertainty"):
        ids.insert(0, "resistance_uncertainty_from_voltage_current")
    seen = set()
    return [x for x in ids if not (x in seen or seen.add(x))]


def solve_numeric_payload(question: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Solve a parser JSON payload with deterministic formulas.

    The payload is intentionally treated as extraction only. If it contains a
    final answer or code, this function rejects it.
    """

    if not isinstance(payload, dict):
        return None
    if any(key in payload for key in ("answer", "unit_answer", "python_code", "golden_code", "final_result")):
        return None
    givens = _givens(payload, question)
    if not givens:
        return None
    out_unit = target_unit(payload)
    if not out_unit:
        role = target_role(payload)
        default_units = {
            "resistance": "ohm",
            "impedance": "ohm",
            "resistance_uncertainty": "ohm",
            "absolute_error": "",
            "current": "A",
            "voltage": "V",
            "power": "W",
            "capacitance": "F",
            "inductance": "H",
            "energy": "J",
            "charge": "C",
            "force": "N",
            "electric_field": "V/m",
            "magnetic_flux": "Wb",
            "magnetic_field": "T",
            "frequency": "Hz",
            "angular_frequency": "rad/s",
            "relative_error": "%",
        }
        out_unit = default_units.get(role, "")
    if not out_unit:
        return None

    for formula_id in _infer_formula_ids(payload, givens):
        try:
            value_si, calc = _solve_formula(formula_id, payload, givens)
        except Exception:
            continue
        if not math.isfinite(value_si):
            continue
        return _result(payload, givens, formula_id, value_si, out_unit, calc)
    return None
