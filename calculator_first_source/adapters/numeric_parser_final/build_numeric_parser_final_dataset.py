from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ALL = ROOT / "adapters" / "numeric_parser_v2" / "dataset" / "numeric_parser_v2_all.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "dataset"

import sys

sys.path.insert(0, str(ROOT))

from src.calculator import solve_numeric_payload
from src.payload_validator import validate_numeric_payload


STRICT_SCHEMA_PROMPT = """
You extract numeric physics problems into JSON for a calculator. Return exactly one JSON object and nothing else.
Do not solve. Do not output answer, unit_answer, cot, python_code, golden_code, or final_result.
Do not put givens, constraints, or formula_candidates inside target.
Every given must include raw_span copied exactly from the question.
Use this schema:
{
  "question_kind": "calculation",
  "topic": "one_topic_string",
  "target": {"symbol": "...", "role": "...", "unit": "...", "raw_span": "..."},
  "givens": [
    {"name": "...", "role": "...", "raw_span": "...", "value_text": "...", "unit_text": "...", "sign": "unknown", "qualifiers": []}
  ],
  "constraints": ["..."],
  "formula_candidates": [{"formula_id": "...", "variable_map": {}, "reason": "..."}],
  "confidence": 0.0
}
""".strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def given(name: str, role: str, raw_span: str, value_text: str, unit_text: str, qualifiers: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "raw_span": raw_span,
        "value_text": value_text,
        "unit_text": unit_text,
        "sign": "unknown",
        "qualifiers": qualifiers or [],
    }


def payload(
    topic: str,
    target_symbol: str,
    target_role: str,
    target_unit: str,
    target_span: str,
    givens: list[dict[str, Any]],
    constraints: list[str],
    formula_id: str,
) -> dict[str, Any]:
    return {
        "question_kind": "calculation",
        "topic": topic,
        "target": {"symbol": target_symbol, "role": target_role, "unit": target_unit, "raw_span": target_span},
        "givens": givens,
        "constraints": constraints,
        "formula_candidates": [
            {
                "formula_id": formula_id,
                "variable_map": {},
                "reason": "Final rebuild coverage row; calculator reverified the extracted quantities and formula.",
            }
        ],
        "confidence": 0.95,
    }


def row(row_id: str, question: str, item: dict[str, Any]) -> dict[str, Any]:
    solved = solve_numeric_payload(question, item)
    if solved is None:
        raise ValueError(f"calculator_failed:{row_id}")
    validation = validate_numeric_payload(question, item)
    if not validation["ok"]:
        raise ValueError(f"validator_failed:{row_id}:{validation}")
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": STRICT_SCHEMA_PROMPT},
            {"role": "user", "content": "Question: " + question},
            {"role": "assistant", "content": json.dumps(item, ensure_ascii=False, separators=(",", ":"))},
        ],
        "metadata": {
            "source": "final_calculator_expansion",
            "source_id": row_id,
            "answer": solved["answer"],
            "unit": solved["unit"],
            "formula_id": solved["formula_id"],
            "topic": solved["topic"],
            "schema_version": "numeric_parser_final_strict_schema",
            "strict_prompt": True,
            "calculator_verified": True,
        },
    }


def expansion_rows() -> list[dict[str, Any]]:
    cases: list[tuple[str, str, dict[str, Any]]] = []

    def add(row_id: str, question: str, item: dict[str, Any]) -> None:
        cases.append((row_id, question, item))

    # Circuit power and resistance rearrangements.
    for idx, (p, r) in enumerate([(72, 8), (45, 5), (128, 32), (200, 50), (98, 2)]):
        add(
            f"NPF_power_I_PR_{idx:03d}",
            f"A resistor dissipates {p} W while its resistance is {r} ohm. What current flows through it?",
            payload("circuit_power", "I", "current", "A", "current", [given("P", "power", f"{p} W", str(p), "W"), given("R", "resistance", f"{r} ohm", str(r), "ohm")], [], "current_from_power_resistance"),
        )
        add(
            f"NPF_power_U_PR_{idx:03d}",
            f"A load consumes {p} W and has resistance {r} ohm. Find the voltage across the load.",
            payload("circuit_power", "U", "voltage", "V", "voltage", [given("P", "power", f"{p} W", str(p), "W"), given("R", "resistance", f"{r} ohm", str(r), "ohm")], [], "voltage_from_power_resistance"),
        )

    # Wire resistance.
    wire_values = [(1.1e-6, 0.8, 4e-6), (1.7e-8, 2.0, 1e-6), (2.8e-8, 5.0, 2e-6), (5e-7, 1.2, 3e-6), (9e-7, 0.6, 1.5e-6)]
    for idx, (rho, length, area) in enumerate(wire_values):
        add(
            f"NPF_wire_R_{idx:03d}",
            f"A uniform wire has resistivity {rho:g} ohm m, length {length:g} m, and cross-sectional area {area:g} m2. Calculate its resistance.",
            payload("circuit_resistance", "R", "resistance", "ohm", "resistance", [given("rho", "resistivity", f"{rho:g} ohm m", f"{rho:g}", "ohm m"), given("l", "length", f"{length:g} m", f"{length:g}", "m"), given("S", "area", f"{area:g} m2", f"{area:g}", "m2")], [], "wire_resistance_from_resistivity_length_area"),
        )

    # Capacitor rearrangements.
    cap_qu = [(60, 12), (25, 5), (180, 30), (84, 7), (150, 50)]
    for idx, (q_uc, u_v) in enumerate(cap_qu):
        add(
            f"NPF_cap_C_QU_{idx:03d}",
            f"A capacitor carries charge {q_uc} uC when the voltage across it is {u_v} V. Determine its capacitance.",
            payload("capacitor", "C", "capacitance", "uF", "capacitance", [given("Q", "charge", f"{q_uc} uC", str(q_uc), "uC"), given("U", "voltage", f"{u_v} V", str(u_v), "V")], [], "capacitor_capacitance_from_charge_voltage"),
        )
        add(
            f"NPF_cap_W_QU_{idx:03d}",
            f"A charged capacitor has charge {q_uc} uC and voltage {u_v} V. Calculate the stored electric energy.",
            payload("capacitor", "W", "energy", "mJ", "energy", [given("Q", "charge", f"{q_uc} uC", str(q_uc), "uC"), given("U", "voltage", f"{u_v} V", str(u_v), "V")], [], "capacitor_energy_from_Q_U"),
        )
    cap_qc = [(40, 8), (120, 6), (75, 15), (90, 3), (55, 11)]
    for idx, (q_uc, c_uf) in enumerate(cap_qc):
        add(
            f"NPF_cap_U_QC_{idx:03d}",
            f"The charge on a capacitor is {q_uc} uC and its capacitance is {c_uf} uF. What is the voltage?",
            payload("capacitor", "U", "voltage", "V", "voltage", [given("Q", "charge", f"{q_uc} uC", str(q_uc), "uC"), given("C", "capacitance", f"{c_uf} uF", str(c_uf), "uF")], [], "capacitor_voltage_from_charge_capacitance"),
        )
        add(
            f"NPF_cap_W_QC_{idx:03d}",
            f"A capacitor stores charge {q_uc} uC with capacitance {c_uf} uF. Find its stored energy.",
            payload("capacitor", "W", "energy", "mJ", "energy", [given("Q", "charge", f"{q_uc} uC", str(q_uc), "uC"), given("C", "capacitance", f"{c_uf} uF", str(c_uf), "uF")], [], "capacitor_energy_from_Q_C"),
        )

    # Parallel plates and dielectric.
    for idx, (eps_r, area_cm2, d_mm) in enumerate([(2.5, 40, 0.5), (4.0, 25, 0.2), (3.2, 60, 0.8), (6.0, 18, 0.3), (1.8, 75, 1.0)]):
        add(
            f"NPF_plate_C_eps_{idx:03d}",
            f"A parallel-plate capacitor has dielectric constant {eps_r:g}, plate area {area_cm2:g} cm2, and separation {d_mm:g} mm. Calculate its capacitance.",
            payload("capacitor", "C", "capacitance", "pF", "capacitance", [given("eps_r", "dielectric_constant", f"{eps_r:g}", f"{eps_r:g}", ""), given("A", "area", f"{area_cm2:g} cm2", f"{area_cm2:g}", "cm2"), given("d", "distance", f"{d_mm:g} mm", f"{d_mm:g}", "mm")], [], "parallel_plate_capacitance_with_dielectric"),
        )
    for idx, (u, d_cm) in enumerate([(120, 3), (450, 5), (36, 0.8), (1000, 2), (24, 1.2)]):
        add(
            f"NPF_plate_E_Ud_{idx:03d}",
            f"Two parallel plates have a potential difference of {u:g} V and are separated by {d_cm:g} cm. Find the electric field between them.",
            payload("electrostatics_field", "E", "electric_field", "V/m", "electric field", [given("U", "voltage", f"{u:g} V", f"{u:g}", "V"), given("d", "distance", f"{d_cm:g} cm", f"{d_cm:g}", "cm")], [], "parallel_plate_field_from_voltage_distance"),
        )

    # Inductor current from energy.
    for idx, (w_mj, l_h) in enumerate([(6.25, 0.5), (18, 0.4), (2.4, 0.3), (50, 2), (1.8, 0.6)]):
        add(
            f"NPF_ind_I_WL_{idx:03d}",
            f"An inductor with inductance {l_h:g} H stores magnetic energy {w_mj:g} mJ. What is the current?",
            payload("induction", "I", "current", "A", "current", [given("L", "inductance", f"{l_h:g} H", f"{l_h:g}", "H"), given("W", "energy", f"{w_mj:g} mJ", f"{w_mj:g}", "mJ")], [], "inductor_current_from_energy_inductance"),
        )

    # Electrostatic/vector geometry.
    for idx, (q_nc, r_cm) in enumerate([(7, 6), (5, 8), (4, 12), (9, 10), (3, 5)]):
        add(
            f"NPF_force_right_{idx:03d}",
            f"Three equal charges of {q_nc:g} nC occupy the corners of a right isosceles triangle. The charge at the right-angle corner is {r_cm:g} cm from each of the other two charges. Find the net force on it.",
            payload("electrostatics_force", "F", "force", "N", "net force", [given("q", "charge", f"{q_nc:g} nC", f"{q_nc:g}", "nC"), given("r", "distance", f"{r_cm:g} cm", f"{r_cm:g}", "cm")], ["right angle", "two equal perpendicular Coulomb forces"], "coulomb_force_right_angle_equal_charges"),
        )
        add(
            f"NPF_force_equilateral_{idx:03d}",
            f"Three equal positive charges of {q_nc:g} nC are placed at the vertices of an equilateral triangle of side {r_cm:g} cm. Find the resultant force on one charge.",
            payload("electrostatics_force", "F", "force", "N", "resultant force", [given("q", "charge", f"{q_nc:g} nC", f"{q_nc:g}", "nC"), given("r", "distance", f"{r_cm:g} cm", f"{r_cm:g}", "cm")], ["equilateral triangle", "60 degrees between equal force components"], "coulomb_force_equilateral_equal_charges"),
        )
    for idx, (q1_nc, q2_nc, r_cm) in enumerate([(3, 4, 9), (6, 2, 8), (5, 5, 10), (7, 1, 12), (2, 8, 6)]):
        add(
            f"NPF_field_two_angle_{idx:03d}",
            f"At a point, charges {q1_nc:g} nC and {q2_nc:g} nC create electric-field directions separated by 60 degrees. Each charge is {r_cm:g} cm from the point. Find the resultant electric field.",
            payload("electrostatics_field", "E", "electric_field", "V/m", "resultant electric field", [given("q1", "charge", f"{q1_nc:g} nC", f"{q1_nc:g}", "nC"), given("q2", "charge", f"{q2_nc:g} nC", f"{q2_nc:g}", "nC"), given("r", "distance", f"{r_cm:g} cm", f"{r_cm:g}", "cm")], ["60 degrees between field vectors"], "electric_field_two_charges_angle"),
        )
    for idx, (a, b) in enumerate([(7, 12), (9, 40), (15, 20), (5, 5), (6, 8)]):
        add(
            f"NPF_force_resultant_{idx:03d}",
            f"Two forces of {a:g} N and {b:g} N act at right angles on the same object. What is the resultant force?",
            payload("general_physics", "F", "force", "N", "resultant force", [given("F1", "force", f"{a:g} N", f"{a:g}", "N"), given("F2", "force", f"{b:g} N", f"{b:g}", "N")], ["right angle", "perpendicular"], "resultant_two_vectors_from_angle"),
        )
        add(
            f"NPF_field_resultant_{idx:03d}",
            f"Two perpendicular electric fields have magnitudes {a * 100:g} V/m and {b * 100:g} V/m. Calculate the resultant electric field.",
            payload("electrostatics_field", "E", "electric_field", "V/m", "resultant electric field", [given("E1", "electric_field", f"{a * 100:g} V/m", f"{a * 100:g}", "V/m"), given("E2", "electric_field", f"{b * 100:g} V/m", f"{b * 100:g}", "V/m")], ["perpendicular"], "electric_field_resultant_two_vectors_from_angle"),
        )

    # AC/RLC.
    for idx, (c_uf, f_hz) in enumerate([(10, 50), (20, 60), (2.5, 1000), (40, 90), (5, 400)]):
        add(
            f"NPF_XC_{idx:03d}",
            f"A capacitor of {c_uf:g} uF is connected to an AC source of frequency {f_hz:g} Hz. Find the capacitive reactance.",
            payload("ac_resonance", "X_C", "resistance", "ohm", "capacitive reactance", [given("C", "capacitance", f"{c_uf:g} uF", f"{c_uf:g}", "uF"), given("f", "frequency", f"{f_hz:g} Hz", f"{f_hz:g}", "Hz")], [], "capacitive_reactance_from_C_f"),
        )
    for idx, (u, r, xl, xc) in enumerate([(100, 30, 40, 0), (120, 50, 90, 30), (220, 80, 100, 40), (60, 20, 50, 10), (36, 12, 20, 5)]):
        giv = [given("U", "voltage", f"{u:g} V", f"{u:g}", "V"), given("R", "resistance", f"{r:g} ohm", f"{r:g}", "ohm"), given("XL", "inductive_reactance", f"{xl:g} ohm", f"{xl:g}", "ohm"), given("XC", "capacitive_reactance", f"{xc:g} ohm", f"{xc:g}", "ohm")]
        add(
            f"NPF_rlc_I_{idx:03d}",
            f"In a series RLC circuit, U = {u:g} V, R = {r:g} ohm, XL = {xl:g} ohm, and XC = {xc:g} ohm. Find the current.",
            payload("ac_resonance", "I", "current", "A", "current", giv, ["series RLC"], "rlc_series_current_from_U_R_XL_XC"),
        )
        add(
            f"NPF_rlc_P_{idx:03d}",
            f"A series RLC circuit has U = {u:g} V, R = {r:g} ohm, XL = {xl:g} ohm, and XC = {xc:g} ohm. Calculate the average power consumed.",
            payload("ac_resonance", "P", "power", "W", "average power", giv, ["series RLC"], "rlc_series_power_from_U_R_XL_XC"),
        )

    # Measurement.
    for idx, (value, rel) in enumerate([(8.0, 2.5), (12.5, 1.2), (100, 0.5), (36, 4.167), (20, 5)]):
        add(
            f"NPF_abs_from_rel_{idx:03d}",
            f"A measured value is {value:g} m and its relative error is {rel:g} %. Find the absolute error.",
            payload("measurement_error", "Delta_x", "absolute_error", "m", "absolute error", [given("x", "measured_value", f"{value:g} m", f"{value:g}", "m"), given("rel", "relative_error", f"{rel:g} %", f"{rel:g}", "%")], [], "absolute_error_from_relative"),
        )
    for idx, lc in enumerate([0.2, 0.5, 1.0, 0.02, 0.1]):
        add(
            f"NPF_abs_least_{idx:03d}",
            f"A ruler has least count {lc:g} cm. Using half the least count as uncertainty, find the absolute error.",
            payload("measurement_error", "Delta_x", "absolute_error", "cm", "absolute error", [given("least_count", "least_count", f"{lc:g} cm", f"{lc:g}", "cm")], ["half least count"], "absolute_error_from_least_count_half"),
        )

    # Underrepresented existing formulas.
    for idx, f_hz in enumerate([25, 50, 60, 400, 1000, 2500]):
        add(
            f"NPF_omega_{idx:03d}",
            f"An oscillator has frequency {f_hz:g} Hz. Calculate its angular frequency.",
            payload("LC_oscillation", "omega", "angular_frequency", "rad/s", "angular frequency", [given("f", "frequency", f"{f_hz:g} Hz", f"{f_hz:g}", "Hz")], [], "angular_frequency_from_frequency"),
        )
    for idx, (c1, c2) in enumerate([(4, 6), (10, 15), (2, 3), (30, 60), (8, 12), (5, 20)]):
        add(
            f"NPF_cap_series_{idx:03d}",
            f"Two capacitors of {c1:g} uF and {c2:g} uF are connected in series. Find the equivalent capacitance.",
            payload("capacitor", "C_eq", "capacitance", "uF", "equivalent capacitance", [given("C1", "capacitance", f"{c1:g} uF", f"{c1:g}", "uF"), given("C2", "capacitance", f"{c2:g} uF", f"{c2:g}", "uF")], ["series"], "capacitance_series"),
        )
    for idx, (force, q_uc) in enumerate([(0.12, 4), (0.45, 9), (2.5, 5), (1.2, 3), (0.08, 2), (3.6, 12)]):
        add(
            f"NPF_E_Fq_{idx:03d}",
            f"A charge of {q_uc:g} uC experiences an electric force of {force:g} N. Determine the electric field strength.",
            payload("electrostatics_field", "E", "electric_field", "V/m", "electric field strength", [given("F", "force", f"{force:g} N", f"{force:g}", "N"), given("q", "charge", f"{q_uc:g} uC", f"{q_uc:g}", "uC")], [], "electric_field_from_force_charge"),
        )
    for idx, (n, current, area_cm2) in enumerate([(800, 2, 5), (1200, 1.5, 8), (2000, 0.8, 3), (1500, 3, 10), (600, 4, 6), (2500, 1.2, 4)]):
        add(
            f"NPF_flux_solenoid_{idx:03d}",
            f"A long solenoid has turn density {n:g} turns/m, current {current:g} A, and cross-sectional area {area_cm2:g} cm2. Find the magnetic flux through one turn.",
            payload("induction", "Phi", "magnetic_flux", "Wb", "magnetic flux", [given("n", "turn_density", f"{n:g} turns/m", f"{n:g}", "turns/m"), given("I", "current", f"{current:g} A", f"{current:g}", "A"), given("S", "area", f"{area_cm2:g} cm2", f"{area_cm2:g}", "cm2")], [], "magnetic_flux_from_solenoid_area"),
        )
    for idx, (p1, p2) in enumerate([(20, 35), (12, 18), (40, 60), (5, 15), (100, 50), (7.5, 12.5)]):
        add(
            f"NPF_power_sum_{idx:03d}",
            f"Two devices dissipate {p1:g} W and {p2:g} W. What is their total power?",
            payload("circuit_power", "P", "power", "W", "total power", [given("P1", "power", f"{p1:g} W", f"{p1:g}", "W"), given("P2", "power", f"{p2:g} W", f"{p2:g}", "W")], [], "power_total_sum"),
        )
    for idx, (u, du, i, di) in enumerate([(6, 0.1, 0.3, 0.01), (12, 0.2, 2, 0.05), (36, 0.6, 2, 0.05), (9, 0.1, 0.45, 0.02), (24, 0.4, 1.2, 0.04), (18, 0.3, 0.9, 0.03)]):
        add(
            f"NPF_dR_UI_{idx:03d}",
            f"A resistance is measured using U = {u:g} V with uncertainty {du:g} V and I = {i:g} A with uncertainty {di:g} A. Find the absolute uncertainty of R.",
            payload("measurement_error", "Delta_R", "resistance_uncertainty", "ohm", "absolute uncertainty", [given("U", "voltage", f"{u:g} V", f"{u:g}", "V", [f"uncertainty {du:g} V"]), given("I", "current", f"{i:g} A", f"{i:g}", "A", [f"uncertainty {di:g} A"])], ["R = U/I", "maximum uncertainty"], "resistance_uncertainty_from_voltage_current"),
        )
    for idx, (power, voltage) in enumerate([(90, 30), (48, 12), (150, 50), (500, 220), (36, 9), (72, 18)]):
        add(
            f"NPF_I_PU_{idx:03d}",
            f"An electrical device has power {power:g} W at voltage {voltage:g} V. Calculate the current.",
            payload("circuit_power", "I", "current", "A", "current", [given("P", "power", f"{power:g} W", f"{power:g}", "W"), given("U", "voltage", f"{voltage:g} V", f"{voltage:g}", "V")], [], "current_from_power_voltage"),
        )
    for idx, (r, xl, xc) in enumerate([(30, 40, 0), (50, 90, 30), (80, 100, 40), (20, 50, 10), (12, 20, 5), (60, 75, 15)]):
        add(
            f"NPF_Z_RLC_{idx:03d}",
            f"A series RLC circuit has R = {r:g} ohm, XL = {xl:g} ohm, and XC = {xc:g} ohm. Calculate its impedance.",
            payload("ac_resonance", "Z", "resistance", "ohm", "impedance", [given("R", "resistance", f"{r:g} ohm", f"{r:g}", "ohm"), given("XL", "inductive_reactance", f"{xl:g} ohm", f"{xl:g}", "ohm"), given("XC", "capacitive_reactance", f"{xc:g} ohm", f"{xc:g}", "ohm")], ["series RLC"], "series_ac_impedance_from_R_XL_XC"),
        )
    for idx, (turns, length, current) in enumerate([(500, 0.25, 2), (1200, 0.6, 1.5), (800, 0.4, 3), (300, 0.15, 4), (2000, 1.0, 0.8), (1500, 0.5, 2.5)]):
        add(
            f"NPF_B_solenoid_Nl_{idx:03d}",
            f"A solenoid has {turns:g} turns over length {length:g} m and carries current {current:g} A. Find the magnetic field inside it.",
            payload("induction", "B", "magnetic_field", "T", "magnetic field", [given("N", "turns", f"{turns:g} turns", f"{turns:g}", "turns"), given("l", "length", f"{length:g} m", f"{length:g}", "m"), given("I", "current", f"{current:g} A", f"{current:g}", "A")], [], "solenoid_field_from_turns_length_current"),
        )
    for idx, (phi1, phi2, dt) in enumerate([(0.006, 0, 0.01), (0.02, 0.005, 0.03), (0.001, 0.004, 0.002), (0.12, 0.02, 0.5), (4e-4, 0, 1e-3), (0.08, 0.03, 0.1)]):
        add(
            f"NPF_faraday_{idx:03d}",
            f"The magnetic flux changes from {phi1:g} Wb to {phi2:g} Wb in {dt:g} s. Find the average induced emf.",
            payload("induction", "e", "voltage", "V", "induced emf", [given("Phi1", "magnetic_flux", f"{phi1:g} Wb", f"{phi1:g}", "Wb"), given("Phi2", "magnetic_flux", f"{phi2:g} Wb", f"{phi2:g}", "Wb"), given("dt", "time", f"{dt:g} s", f"{dt:g}", "s")], [], "faraday_emf_from_flux_change"),
        )
    for idx, (emf, i1, i2, dt) in enumerate([(6, 1, 3, 0.2), (12, 0, 4, 0.5), (2.5, 1.5, 0.5, 0.1), (18, 2, 8, 0.3), (0.8, 0.1, 0.5, 0.04), (24, 3, 9, 0.2)]):
        add(
            f"NPF_L_from_emf_{idx:03d}",
            f"A coil has induced emf {emf:g} V when its current changes from {i1:g} A to {i2:g} A in {dt:g} s. Find the inductance.",
            payload("induction", "L", "inductance", "H", "inductance", [given("e", "voltage", f"{emf:g} V", f"{emf:g}", "V"), given("I1", "current", f"{i1:g} A", f"{i1:g}", "A"), given("I2", "current", f"{i2:g} A", f"{i2:g}", "A"), given("dt", "time", f"{dt:g} s", f"{dt:g}", "s")], [], "inductance_from_emf_current_change"),
        )
    for idx, (l_h, f_hz) in enumerate([(0.1, 60), (0.25, 50), (0.5, 71), (0.03, 400), (0.8, 25), (0.002, 1000)]):
        add(
            f"NPF_XL_{idx:03d}",
            f"An inductor of {l_h:g} H is used at frequency {f_hz:g} Hz. Find the inductive reactance.",
            payload("ac_resonance", "X_L", "resistance", "ohm", "inductive reactance", [given("L", "inductance", f"{l_h:g} H", f"{l_h:g}", "H"), given("f", "frequency", f"{f_hz:g} Hz", f"{f_hz:g}", "Hz")], [], "inductive_reactance_from_L_f"),
        )
    for idx, (b_mt, area_cm2) in enumerate([(0.35, 12), (2, 3), (1.5, 20), (0.8, 15), (5, 2), (0.12, 50)]):
        add(
            f"NPF_flux_BS_{idx:03d}",
            f"A flat coil surface of area {area_cm2:g} cm2 is aligned with a uniform magnetic field of {b_mt:g} mT. Find the magnetic flux.",
            payload("induction", "Phi", "magnetic_flux", "Wb", "magnetic flux", [given("B", "magnetic_field", f"{b_mt:g} mT", f"{b_mt:g}", "mT"), given("S", "area", f"{area_cm2:g} cm2", f"{area_cm2:g}", "cm2")], ["normal aligned with field"], "magnetic_flux_from_B_area_angle"),
        )

    rows = []
    for row_id, question, item in cases:
        rows.append(row(row_id, question, item))
    return rows


def normalize_legacy_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    if messages:
        messages[0] = {"role": "system", "content": STRICT_SCHEMA_PROMPT}
    metadata = dict(row.get("metadata") or {})
    metadata["schema_version"] = "numeric_parser_final_strict_schema"
    metadata["strict_prompt"] = True
    row["messages"] = messages
    row["metadata"] = metadata
    return row


def main() -> None:
    if not SOURCE_ALL.exists():
        existing = OUT_DIR / "numeric_parser_final_all.jsonl"
        if existing.exists():
            print(
                json.dumps(
                    {
                        "status": "already_built",
                        "message": "The old numeric_parser_v2 source folder has been removed. Use the existing final dataset files in this folder for training.",
                        "dataset": str(existing),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
        raise FileNotFoundError(f"Missing source dataset: {SOURCE_ALL}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    legacy = [normalize_legacy_row(dict(row)) for row in read_jsonl(SOURCE_ALL)]
    expansion = expansion_rows()

    by_id: dict[str, dict[str, Any]] = {}
    for item in [*legacy, *expansion]:
        by_id[item["id"]] = item
    rows = list(by_id.values())
    random.Random(20260614).shuffle(rows)
    valid_size = max(220, int(len(rows) * 0.12))
    valid = rows[:valid_size]
    train = rows[valid_size:]

    write_jsonl(OUT_DIR / "numeric_parser_final_all.jsonl", rows)
    write_jsonl(OUT_DIR / "numeric_parser_final_train.jsonl", train)
    write_jsonl(OUT_DIR / "numeric_parser_final_valid.jsonl", valid)

    formula_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for item in rows:
        meta = item.get("metadata") or {}
        formula = str(meta.get("formula_id") or "")
        source = str(meta.get("source") or "")
        if formula:
            formula_counts[formula] = formula_counts.get(formula, 0) + 1
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1

    generated_log = [
        {
            "id": item["id"],
            "formula_id": item["metadata"]["formula_id"],
            "answer": item["metadata"]["answer"],
            "unit": item["metadata"]["unit"],
            "question": item["messages"][1]["content"].replace("Question: ", ""),
        }
        for item in expansion
    ]
    (OUT_DIR / "numeric_parser_final_expansion_log.json").write_text(json.dumps(generated_log, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "total_rows": len(rows),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "legacy_rows": len(legacy),
        "expansion_rows": len(expansion),
        "source_dataset": str(SOURCE_ALL),
        "schema_version": "numeric_parser_final_strict_schema",
        "calculator_formula_count": 69,
        "policy": "Final numeric parser dataset keeps calculator-verified strict rows and adds calculator-reverified coverage rows for newly added formula families. Assistant output remains extraction-only: no answer, CoT, Python code, golden_code, or final_result.",
        "source_counts": source_counts,
        "formula_counts": dict(sorted(formula_counts.items(), key=lambda x: (-x[1], x[0]))),
    }
    (OUT_DIR / "numeric_parser_final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
