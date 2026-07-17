from __future__ import annotations

import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
OUT_DIR = Path(__file__).resolve().parent / "dataset"
SOURCE_CSV = WORKSPACE / "submissions" / "current" / "physics_api_package" / "data" / "verified_golden_expanded.csv"
SEED_DIR = ROOT / "adapters" / "chlt_reasoner" / "dataset"

SYSTEM_PROMPT = (
    "You answer conceptual physics questions. Return exactly one JSON object. "
    "If options are present, the answer must match one option. If uncertain, answer Uncertain. Do not output code."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def normalize_text(text: Any) -> str:
    value = str(text or "")
    value = value.replace("μ", "u").replace("µ", "u").replace("Ω", "ohm")
    value = value.replace("π", "pi").replace("×", "x").replace("−", "-")
    return re.sub(r"\s+", " ", value.strip())


def parse_number(text: Any) -> float | None:
    value = normalize_text(text)
    value = re.sub(r"([+-]?\d+(?:\.\d+)?)\s*(?:x|\*)\s*10\^?([+-]?\d+)", r"\1e\2", value, flags=re.I)
    match = re.search(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", value, flags=re.I)
    return float(match.group(0)) if match else None


def extract_rlc_values(question: str) -> tuple[float | None, float | None, float | None]:
    q = normalize_text(question)
    l_match = re.search(r"\bL\s*=\s*([+-]?\d+(?:\.\d+)?)\s*(mH|uH|H)\b", q, flags=re.I)
    c_match = re.search(r"\bC\s*=\s*([+-]?\d+(?:\.\d+)?)\s*(uF|nF|pF|mF|F)\b", q, flags=re.I)
    f_match = re.search(r"\bf\s*=\s*([+-]?\d+(?:\.\d+)?)\s*(Hz|kHz)\b", q, flags=re.I)
    if not l_match:
        l_match = re.search(r"induct(?:or|ance)?(?:\s+of|\s+is|\s*=)?\s*([+-]?\d+(?:\.\d+)?)\s*(mH|uH|H)\b", q, flags=re.I)
    if not c_match:
        c_match = re.search(r"capacit(?:or|ance)?(?:\s+of|\s+is|\s*=)?\s*([+-]?\d+(?:\.\d+)?)\s*(uF|nF|pF|mF|F)\b", q, flags=re.I)
    if not f_match:
        f_match = re.search(r"frequency(?:\s+of|\s+is|\s*=)?\s*([+-]?\d+(?:\.\d+)?)\s*(Hz|kHz)\b", q, flags=re.I)
    if not f_match:
        hz_matches = list(re.finditer(r"([+-]?\d+(?:\.\d+)?)\s*(Hz|kHz)\b", q, flags=re.I))
        if hz_matches:
            f_match = hz_matches[-1]

    def scale_l(value: float, unit: str) -> float:
        unit = unit.lower()
        return value * {"h": 1.0, "mh": 1e-3, "uh": 1e-6}[unit]

    def scale_c(value: float, unit: str) -> float:
        unit = unit.lower()
        return value * {"f": 1.0, "mf": 1e-3, "uf": 1e-6, "nf": 1e-9, "pf": 1e-12}[unit]

    def scale_f(value: float, unit: str) -> float:
        return value * (1000.0 if unit.lower() == "khz" else 1.0)

    L = scale_l(float(l_match.group(1)), l_match.group(2)) if l_match else None
    C = scale_c(float(c_match.group(1)), c_match.group(2)) if c_match else None
    f = scale_f(float(f_match.group(1)), f_match.group(2)) if f_match else None
    return L, C, f


def rlc_resonance_answer(question: str) -> tuple[str, list[str], float] | None:
    L, C, f = extract_rlc_values(question)
    if not (L and C and f):
        return None
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
    relative = abs(f - f0) / max(f0, 1e-12)
    answer = "Yes" if relative <= 0.015 else "No"
    evidence = [
        f"Series RLC resonance occurs when the operating frequency equals f0 = 1/(2*pi*sqrt(LC)).",
        f"Using the extracted values gives f0 = {f0:.3g} Hz, while the operating frequency is {f:.3g} Hz.",
        "Therefore the circuit is in resonance." if answer == "Yes" else "Therefore the circuit is not in resonance.",
    ]
    confidence = 0.96 if relative <= 0.005 or relative >= 0.05 else 0.9
    return answer, evidence, confidence


def output_obj(topic: str, concept: str, answer_type: str, answer: str, evidence: list[str], confidence: float) -> dict[str, Any]:
    return {
        "question_kind": "conceptual",
        "topic": topic,
        "concept": concept,
        "answer_type": answer_type,
        "answer": answer,
        "evidence": evidence[:3],
        "confidence": round(float(confidence), 3),
    }


def make_row(row_id: str, question: str, obj: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Question: " + normalize_text(question)},
            {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False, separators=(",", ":"))},
        ],
        "metadata": {
            "source": source,
            "topic": obj["topic"],
            "concept": obj["concept"],
            "answer": obj["answer"],
            "schema_version": "chlt_reasoner_final",
        },
    }


def build_verified_rows() -> list[dict[str, Any]]:
    rows = []
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rid = str(raw.get("id") or "")
            if not rid.startswith("CHLT"):
                continue
            question = str(raw.get("question") or "")
            answer = str(raw.get("answer") or "").strip()
            verified = rlc_resonance_answer(question)
            if verified is not None:
                calc_answer, evidence, conf = verified
                # Prefer the official locked answer, but lower confidence if it
                # disagrees with the independent resonance check.
                if answer not in {"Yes", "No"}:
                    answer = calc_answer
                elif answer != calc_answer:
                    conf = 0.78
                    evidence.append(f"The source label is {answer}; this row is kept as source-supervised.")
            else:
                evidence = [
                    "This is a conceptual physics question.",
                    "The answer is taken from the verified CHLT source row.",
                ]
                conf = 0.88
            obj = output_obj("ac_resonance", "series_rlc_resonance_condition", "yes_no", answer, evidence, conf)
            rows.append(make_row("CHLTF_" + rid, question, obj, "verified_golden_expanded_chlt"))
    return rows


def build_seed_rows() -> list[dict[str, Any]]:
    rows = []
    for path in [SEED_DIR / "train.seed.jsonl", SEED_DIR / "valid.seed.jsonl"]:
        for item in read_jsonl(path):
            messages = item.get("messages") or []
            if len(messages) < 3:
                continue
            try:
                obj = json.loads(messages[-1]["content"])
            except Exception:
                continue
            question = str(messages[1].get("content") or "").replace("Question:", "").strip()
            rows.append(make_row("CHLTF_SEED_" + str(item.get("id") or len(rows)), question, obj, "seed_conceptual"))
    return rows


def build_synthetic_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs = [
        (0.5, 20, 50.33), (0.2, 50, 50.33), (0.1, 25, 100.66), (0.4, 10, 79.58),
        (0.3, 30, 53.05), (0.08, 80, 62.91), (1.0, 1, 159.15), (0.25, 100, 31.83),
        (0.5, 20, 40), (0.2, 50, 35), (0.1, 25, 80), (0.4, 10, 60),
        (0.3, 30, 100), (0.08, 80, 100), (1.0, 1, 120), (0.25, 100, 50),
    ]
    templates = [
        "A series RLC circuit has R = {R} ohm, L = {L} H, and C = {C} uF. If the source frequency is {f} Hz, does resonance occur?",
        "For a series AC circuit with L = {L} H and C = {C} uF, is an operating frequency of {f} Hz the resonance condition?",
        "An RLC branch uses an inductor of {L} H and a capacitor of {C} uF. Decide whether {f} Hz is its resonant frequency.",
        "Does a circuit with R = {R} ohm, L = {L} H, and C = {C} uF reach electrical resonance at {f} Hz?",
    ]
    rows = []
    log = []
    idx = 0
    for L, C, f in configs:
        for template in templates:
            R = [10, 20, 30, 40, 50, 60][idx % 6]
            question = template.format(R=R, L=f"{L:g}", C=f"{C:g}", f=f"{f:g}")
            answer, evidence, conf = rlc_resonance_answer(question) or ("Uncertain", ["Could not parse resonance quantities."], 0.3)
            obj = output_obj("ac_resonance", "series_rlc_resonance_condition", "yes_no", answer, evidence, conf)
            row_id = f"CHLTF_SYN_RLC_{idx:04d}"
            rows.append(make_row(row_id, question, obj, "synthetic_verified_rlc_resonance"))
            log.append({"id": row_id, "L_H": L, "C_uF": C, "f_Hz": f, "answer": answer, "question": question})
            idx += 1
    concept_cases = [
        ("In an ideal LC circuit with no resistance, is total electromagnetic energy conserved?", "LC_oscillation", "energy_conservation_in_ideal_lc", "yes_no", "Yes", ["In an ideal LC circuit, energy alternates between electric and magnetic forms.", "With no resistance, there is no dissipative loss, so total electromagnetic energy is conserved."]),
        ("At resonance in a series RLC circuit, is the impedance purely resistive?", "ac_resonance", "impedance_at_resonance", "yes_no", "Yes", ["At resonance, XL equals XC.", "The net reactance is zero, leaving only resistance."]),
        ("For an isolated charged capacitor, what happens to voltage if capacitance decreases?", "capacitor", "isolated_capacitor_charge_constant", "text", "The voltage increases.", ["For an isolated capacitor, charge remains constant.", "Since U = Q/C, decreasing C increases U."]),
        ("When a dielectric is inserted into a connected parallel-plate capacitor, does capacitance increase?", "capacitor", "dielectric_effect_on_capacitance", "yes_no", "Yes", ["A dielectric increases permittivity.", "For parallel plates, C = eps_r eps0 A/d, so larger eps_r means larger capacitance."]),
        ("Does magnetic flux depend on the angle between the magnetic field and the surface normal?", "induction", "magnetic_flux_angle_dependence", "yes_no", "Yes", ["Magnetic flux is Phi = B S cos(theta).", "The cosine factor depends on the angle to the surface normal."]),
        ("If a conductor loop has no change in magnetic flux, is an induced emf produced?", "induction", "faraday_flux_change_condition", "yes_no", "No", ["Faraday's law depends on the rate of change of magnetic flux.", "If the flux is constant, the induced emf is zero."]),
        ("Are electric field vectors added as scalars or as vectors?", "electrostatics_field", "field_superposition_vector", "text", "They are added as vectors.", ["Electric fields obey superposition.", "Direction matters, so components must be combined vectorially."]),
        ("Between two equal positive charges, where is the electric field zero?", "electrostatics_field", "field_cancellation_symmetry", "text", "At the midpoint between the charges.", ["The fields from equal positive charges have equal magnitude at the midpoint.", "Their directions are opposite there, so they cancel."]),
        ("Does a resistor dissipate real power in an AC circuit?", "circuit_power", "resistor_real_power", "yes_no", "Yes", ["A resistor converts electrical energy into heat.", "Real power in a resistor can be written as P = I^2 R."]),
        ("At resonance, does a series RLC circuit have maximum current for a fixed source voltage?", "ac_resonance", "maximum_current_at_resonance", "yes_no", "Yes", ["At resonance the impedance is minimum and equals R.", "For fixed voltage, smaller impedance gives larger current."]),
    ]
    for j, (question, topic, concept, answer_type, answer, evidence) in enumerate(concept_cases):
        obj = output_obj(topic, concept, answer_type, answer, evidence, 0.94)
        row_id = f"CHLTF_SYN_CONCEPT_{j:04d}"
        rows.append(make_row(row_id, question, obj, "synthetic_verified_conceptual"))
        log.append({"id": row_id, "answer": answer, "question": question})
    return rows, log


def build_curated_concept_bank_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []

    def add(
        topic: str,
        concept: str,
        answer_type: str,
        answer: str,
        evidence: list[str],
        questions: list[str],
        confidence: float = 0.94,
    ) -> None:
        for question in questions:
            row_id = f"CHLTF_BANK_{len(rows):04d}"
            obj = output_obj(topic, concept, answer_type, answer, evidence, confidence)
            rows.append(make_row(row_id, question, obj, "synthetic_curated_chlt_concept_bank"))
            log.append({
                "id": row_id,
                "topic": topic,
                "concept": concept,
                "answer_type": answer_type,
                "answer": answer,
                "question": question,
                "verification": "curated physics concept rule",
            })

    add("ac_resonance", "series_rlc_resonance_condition", "yes_no", "Yes",
        ["A series RLC circuit is in resonance when the inductive and capacitive reactances are equal.", "At that point the net reactance is zero."],
        [
            "In a series RLC circuit, does resonance occur when the inductive reactance equals the capacitive reactance?",
            "If XL and XC have the same magnitude in a series AC circuit, should the circuit be considered resonant?",
            "True or false: a series RLC branch resonates when XL = XC.",
        ])
    add("ac_resonance", "impedance_at_resonance", "text", "It is equal to the resistance.",
        ["At resonance, XL - XC = 0.", "The impedance of a series RLC circuit then reduces to the resistance R."],
        [
            "What does the total impedance become at resonance in a series RLC circuit?",
            "When an RLC series circuit is resonant, what part of the impedance remains?",
            "At resonance, is the total impedance reactive or simply resistive?",
        ])
    add("ac_resonance", "power_factor_at_resonance", "yes_no", "Yes",
        ["At resonance the voltage and current are in phase.", "The power factor of a series RLC circuit is therefore one."],
        [
            "At resonance in a series RLC circuit, is the power factor equal to one?",
            "Does a resonant series AC circuit have voltage and current in phase?",
            "True or false: the phase angle is zero at series resonance.",
        ])
    add("ac_resonance", "current_maximum_at_resonance", "yes_no", "Yes",
        ["For a fixed source voltage, current is largest when impedance is smallest.", "At resonance, the series RLC impedance is minimum and equals R."],
        [
            "Does the current reach a maximum at resonance for a fixed-voltage series RLC circuit?",
            "In a series RLC circuit, why is the current largest at resonance?",
            "True or false: the RMS current is maximized at series resonance if the source voltage is fixed.",
        ])
    add("ac_resonance", "resonant_frequency_independent_of_R", "yes_no", "No",
        ["The ideal resonance frequency is f0 = 1/(2*pi*sqrt(LC)).", "Resistance affects damping and current size, not the ideal value of f0."],
        [
            "Does changing the resistor change the ideal resonant frequency of a series RLC circuit?",
            "If only R is changed while L and C stay fixed, should f0 change?",
            "True or false: the ideal resonance frequency depends directly on R.",
        ])
    add("ac_resonance", "below_resonance_capacitive", "text", "The circuit is capacitive.",
        ["Below resonance, XL is smaller than XC.", "The net reactance XL - XC is negative, so the circuit behaves capacitively."],
        [
            "If a series RLC circuit operates below its resonant frequency, is its net behavior inductive or capacitive?",
            "When f is less than f0 in a series RLC circuit, what kind of reactance dominates?",
            "At frequencies below resonance, does the capacitor or inductor dominate the series RLC reactance?",
        ])
    add("ac_resonance", "above_resonance_inductive", "text", "The circuit is inductive.",
        ["Above resonance, XL is larger than XC.", "The net reactance is positive, so the circuit behaves inductively."],
        [
            "If a series RLC circuit is driven above resonance, is the net behavior inductive or capacitive?",
            "When f is greater than f0 in a series RLC circuit, which reactance dominates?",
            "At frequencies above resonance, does the circuit behave more like an inductor?",
        ])
    add("ac_resonance", "resistor_voltage_at_resonance", "yes_no", "Yes",
        ["At resonance, the current is in phase with the source voltage.", "The net reactive voltage cancels, so the resistor voltage equals the source voltage in the ideal series model."],
        [
            "At resonance, is the RMS voltage across the resistor equal to the source RMS voltage in an ideal series RLC circuit?",
            "When UL and UC cancel in a resonant series RLC circuit, does UR match the applied voltage?",
            "True or false: at series resonance, the source voltage appears across the resistor as the net RMS voltage.",
        ])
    add("ac_resonance", "inductor_capacitor_voltage_cancellation", "yes_no", "Yes",
        ["At resonance, UL and UC have equal magnitudes.", "They are opposite in phase, so their net reactive contribution cancels."],
        [
            "At resonance, do the inductor and capacitor voltages cancel in the phasor sum?",
            "In a resonant series RLC circuit, are UL and UC opposite in phase?",
            "True or false: the reactive voltages cancel each other at resonance.",
        ])

    add("capacitor", "capacitance_geometry_dependence", "yes_no", "No",
        ["For an ideal capacitor, capacitance depends on geometry and dielectric material.", "It does not depend on the applied voltage itself."],
        [
            "Does the capacitance of an ideal capacitor depend on the voltage connected to it?",
            "If a capacitor is ideal, does raising the voltage change its capacitance?",
            "True or false: capacitance is determined by geometry and dielectric, not by voltage.",
        ])
    add("capacitor", "parallel_capacitors_add", "yes_no", "Yes",
        ["Parallel capacitors share the same voltage.", "Their charges add, so equivalent capacitance is the sum of capacitances."],
        [
            "For capacitors in parallel, is the equivalent capacitance the sum of the individual capacitances?",
            "When two capacitors are connected in parallel, do their capacitances add directly?",
            "True or false: C_parallel = C1 + C2 for two capacitors.",
        ])
    add("capacitor", "series_capacitance_smaller", "yes_no", "Yes",
        ["For series capacitors, reciprocal capacitances add.", "The equivalent capacitance is smaller than the smallest individual capacitance."],
        [
            "Is the equivalent capacitance of capacitors in series smaller than each individual capacitor?",
            "When capacitors are in series, can the equivalent capacitance be less than the smallest one?",
            "True or false: adding capacitors in series lowers the equivalent capacitance.",
        ])
    add("capacitor", "dielectric_increases_capacitance", "yes_no", "Yes",
        ["A dielectric increases the permittivity between plates.", "For parallel plates, C is proportional to permittivity."],
        [
            "Does inserting a dielectric between capacitor plates increase the capacitance?",
            "If dielectric constant is greater than one, does the capacitance become larger?",
            "True or false: a dielectric slab increases the capacitance of a parallel-plate capacitor.",
        ])
    add("capacitor", "plate_distance_effect", "text", "The capacitance decreases.",
        ["For parallel plates, C = eps*A/d.", "Increasing plate separation d decreases capacitance."],
        [
            "What happens to a parallel-plate capacitor's capacitance if the plate separation is increased?",
            "If the distance between plates doubles while area is fixed, does capacitance increase or decrease?",
            "In a parallel-plate capacitor, how does larger spacing affect C?",
        ])
    add("capacitor", "isolated_capacitor_voltage_change", "text", "The voltage increases.",
        ["For an isolated capacitor, charge stays constant.", "Since U = Q/C, decreasing capacitance increases voltage."],
        [
            "An isolated charged capacitor has its plate separation increased. What happens to the voltage?",
            "If charge is fixed and capacitance decreases, how does the capacitor voltage change?",
            "For an isolated capacitor, does voltage rise when C becomes smaller?",
        ])
    add("capacitor", "battery_connected_voltage_fixed", "yes_no", "Yes",
        ["A connected ideal battery fixes the voltage across the capacitor.", "Changing capacitance changes charge, while voltage remains set by the battery."],
        [
            "If a capacitor remains connected to an ideal battery, is its voltage fixed by the battery?",
            "When dielectric is inserted while the battery stays connected, does the voltage remain the battery voltage?",
            "True or false: a connected voltage source holds capacitor voltage constant.",
        ])
    add("capacitor", "capacitor_energy_formula", "yes_no", "Yes",
        ["Capacitor energy can be written W = 1/2 C U^2.", "It can also be expressed as Q^2/(2C) or 1/2 Q U."],
        [
            "Is the stored energy of a capacitor given by one half C times U squared?",
            "Can capacitor energy be expressed as W = 1/2 C U^2?",
            "True or false: W = 1/2 C U^2 is a capacitor energy relation.",
        ])

    add("induction", "faraday_flux_change_condition", "yes_no", "No",
        ["Faraday's law says induced emf depends on the rate of change of magnetic flux.", "If flux does not change, the induced emf is zero."],
        [
            "If magnetic flux through a loop is constant, is an induced emf produced?",
            "Can a stationary loop in a steady magnetic field have induced emf just because flux exists?",
            "True or false: unchanging magnetic flux is enough to induce emf.",
        ])
    add("induction", "faster_flux_change_larger_emf", "yes_no", "Yes",
        ["The magnitude of induced emf is proportional to |Delta Phi|/Delta t.", "A faster change means a larger rate of flux change."],
        [
            "Does changing magnetic flux more quickly produce a larger induced emf?",
            "If the same flux change happens in less time, is the induced voltage larger?",
            "True or false: induced emf grows when the flux changes faster.",
        ])
    add("induction", "lenz_law_opposes_change", "yes_no", "Yes",
        ["Lenz's law gives the direction of induced current.", "The induced effect opposes the change in magnetic flux that caused it."],
        [
            "Does Lenz's law say the induced current opposes the change that creates it?",
            "When flux through a loop increases, does the induced current act against that increase?",
            "True or false: Lenz's law is an opposition rule for induced current direction.",
        ])
    add("induction", "magnetic_flux_angle_dependence", "yes_no", "Yes",
        ["Magnetic flux is Phi = B S cos(theta).", "The angle is measured between the magnetic field and the surface normal."],
        [
            "Does magnetic flux through a flat surface depend on the angle between B and the surface normal?",
            "If a coil is rotated in a magnetic field, can the magnetic flux change?",
            "True or false: Phi = B S cos theta includes an angle factor.",
        ])
    add("induction", "coil_turns_effect", "yes_no", "Yes",
        ["For a coil, total induced emf scales with the number of turns.", "Faraday's law gives |e| = N |Delta Phi|/Delta t."],
        [
            "Does increasing the number of turns in a coil increase the induced emf for the same flux change per turn?",
            "If two coils have the same flux change per turn, does the coil with more turns have larger induced emf?",
            "True or false: induced emf is proportional to the number of turns.",
        ])
    add("induction", "solenoid_field_current_dependence", "yes_no", "Yes",
        ["For a long solenoid, B = mu n I.", "Increasing current increases the magnetic field inside."],
        [
            "Inside a long solenoid, does the magnetic field increase when current increases?",
            "Is the magnetic field of an ideal solenoid proportional to current?",
            "True or false: B = mu n I for a long solenoid.",
        ])
    add("induction", "inductor_energy_current", "yes_no", "Yes",
        ["Energy in an inductor is W = 1/2 L I^2.", "It is stored in the magnetic field."],
        [
            "Is the magnetic energy stored in an inductor proportional to the square of current?",
            "Can inductor energy be written as W = 1/2 L I^2?",
            "True or false: an inductor stores energy in its magnetic field.",
        ])

    add("LC_oscillation", "lc_energy_exchange", "yes_no", "Yes",
        ["In an ideal LC circuit, energy oscillates between the capacitor's electric field and the inductor's magnetic field.", "With no resistance, total electromagnetic energy is conserved."],
        [
            "In an ideal LC circuit, does energy move back and forth between the capacitor and inductor?",
            "Does an LC oscillator exchange electric-field energy and magnetic-field energy?",
            "True or false: total energy is conserved in an ideal LC oscillator.",
        ])
    add("LC_oscillation", "capacitor_voltage_max_current_zero", "yes_no", "Yes",
        ["When capacitor voltage is maximum, the electric field energy is maximum.", "At that instant the current is zero in an ideal LC oscillator."],
        [
            "In an ideal LC circuit, is the current zero when capacitor voltage is maximum?",
            "At maximum capacitor charge in an LC oscillator, does the inductor current vanish?",
            "True or false: maximum capacitor voltage occurs when current is zero in ideal LC motion.",
        ])
    add("LC_oscillation", "current_max_voltage_zero", "yes_no", "Yes",
        ["When current is maximum, magnetic energy is maximum.", "At that instant the capacitor voltage is zero in the ideal LC model."],
        [
            "In an ideal LC circuit, is capacitor voltage zero when current is maximum?",
            "At maximum inductor current, has the capacitor discharged in an ideal LC oscillator?",
            "True or false: maximum current corresponds to zero capacitor voltage in ideal LC oscillation.",
        ])
    add("LC_oscillation", "lc_frequency_depends_lc", "yes_no", "No",
        ["The ideal angular frequency is omega = 1/sqrt(LC).", "It does not depend on the initial voltage amplitude."],
        [
            "Does increasing the initial capacitor voltage change the ideal LC oscillation frequency?",
            "For an ideal LC oscillator, does frequency depend on amplitude?",
            "True or false: LC frequency is set by L and C rather than the starting voltage.",
        ])

    add("electrostatics_force", "coulomb_force_sign_direction", "yes_no", "Yes",
        ["Like charges repel and unlike charges attract.", "The sign of the charges determines force direction, while magnitude uses absolute values."],
        [
            "Do two charges with the same sign repel each other?",
            "If two charges have opposite signs, is the electric force attractive?",
            "True or false: charge sign determines whether Coulomb force is attractive or repulsive.",
        ])
    add("electrostatics_force", "coulomb_inverse_square", "yes_no", "Yes",
        ["Coulomb force magnitude is proportional to 1/r^2.", "Doubling distance reduces the force to one fourth."],
        [
            "Does Coulomb force decrease with the square of distance?",
            "If the distance between two point charges doubles, does the force become one fourth?",
            "True or false: electric force between point charges follows an inverse-square law.",
        ])
    add("electrostatics_field", "electric_field_direction_positive_test_charge", "yes_no", "Yes",
        ["Electric field direction is defined as the direction of force on a positive test charge.", "A negative charge would feel force opposite to the field."],
        [
            "Is the electric field direction defined by the force on a positive test charge?",
            "Does a positive test charge move in the direction of the electric field force?",
            "True or false: field direction is based on a positive test charge.",
        ])
    add("electrostatics_field", "field_superposition_vector", "text", "They are added as vectors.",
        ["Electric fields obey superposition.", "Since fields have direction, their components must be combined vectorially."],
        [
            "When several charges create electric fields at a point, how are the fields combined?",
            "Are electric fields from different sources added as vectors or ordinary scalars?",
            "In superposition of electric fields, should direction be included?",
        ])
    add("electrostatics_field", "zero_field_equal_like_charges", "text", "At the midpoint between the charges.",
        ["For two equal like charges, the fields at the midpoint have equal magnitudes.", "Their directions are opposite there, so they cancel."],
        [
            "Where is the electric field zero between two equal positive point charges?",
            "For two identical like charges on a line, where do their electric fields cancel?",
            "Between equal same-sign charges, which point has zero net electric field?",
        ])
    add("electrostatics_field", "conductor_field_inside_static", "yes_no", "No",
        ["In electrostatic equilibrium, the electric field inside a conductor is zero.", "Free charges rearrange until the internal field cancels."],
        [
            "Can a conductor in electrostatic equilibrium have a nonzero electric field inside its material?",
            "Inside a conductor at electrostatic equilibrium, is the electric field present?",
            "True or false: the electrostatic field inside a conductor is generally nonzero.",
        ])

    add("circuit_resistance", "series_resistors_add", "yes_no", "Yes",
        ["Series resistors carry the same current.", "Their voltage drops add, so equivalent resistance is the sum."],
        [
            "Do resistors in series add directly?",
            "For a string of series resistors, is total resistance R1 + R2 + ...?",
            "True or false: equivalent resistance increases when another resistor is added in series.",
        ])
    add("circuit_resistance", "parallel_resistance_smaller", "yes_no", "Yes",
        ["Parallel branches provide additional current paths.", "The equivalent resistance is smaller than the smallest branch resistance."],
        [
            "Is the equivalent resistance of parallel resistors smaller than the smallest individual resistor?",
            "Does adding a resistor in parallel reduce total resistance?",
            "True or false: parallel connection lowers equivalent resistance.",
        ])
    add("circuit_power", "resistor_real_power", "yes_no", "Yes",
        ["A resistor dissipates electrical energy as heat.", "Its power can be written P = I^2 R or P = U^2/R."],
        [
            "Does a resistor consume real power in a circuit?",
            "Is electrical energy converted to heat in a resistor?",
            "True or false: P = I squared R describes resistor power.",
        ])
    add("circuit_power", "ideal_capacitor_average_power", "yes_no", "No",
        ["An ideal capacitor stores and returns energy each cycle.", "Its average real power over a complete AC cycle is zero."],
        [
            "Does an ideal capacitor consume average real power over a full AC cycle?",
            "In ideal AC analysis, does a pure capacitor dissipate real power?",
            "True or false: an ideal capacitor has nonzero average power loss.",
        ])
    add("circuit_power", "ideal_inductor_average_power", "yes_no", "No",
        ["An ideal inductor stores and returns magnetic energy.", "Its average real power over a complete AC cycle is zero."],
        [
            "Does an ideal inductor dissipate average real power?",
            "For a pure ideal inductor in AC, is average real power consumed?",
            "True or false: an ideal inductor loses real power every cycle.",
        ])

    add("measurement_error", "relative_error_definition", "yes_no", "Yes",
        ["Relative error is absolute error divided by the measured value.", "It is often multiplied by 100 percent when reported as a percentage."],
        [
            "Is relative error equal to absolute error divided by the measured value?",
            "To express relative error as a percent, do we multiply the ratio by 100 percent?",
            "True or false: relative error compares the absolute uncertainty with the measured value.",
        ])
    add("measurement_error", "sum_error_addition_rule", "yes_no", "Yes",
        ["For sums and differences, absolute uncertainties are added in the usual school-level rule.", "This gives a conservative uncertainty bound."],
        [
            "When adding measured quantities, are their absolute errors usually added?",
            "For A plus B with measurement uncertainties, does the absolute error combine from the absolute errors?",
            "True or false: absolute errors are used directly for addition and subtraction.",
        ])
    add("measurement_error", "product_error_relative_rule", "yes_no", "Yes",
        ["For products and quotients, relative errors are added in the common propagation rule.", "This is why resistance R = U/I uses relative errors of U and I."],
        [
            "For a product or quotient of measured quantities, do relative errors add in the standard rule?",
            "When computing R = U/I, should the relative errors of U and I be combined?",
            "True or false: multiplication and division use relative uncertainty propagation.",
        ])
    add("measurement_error", "least_count_half_rule", "yes_no", "Yes",
        ["For many analog instruments, a common absolute reading uncertainty is half the smallest division.", "This is a measurement convention, not a new physics law."],
        [
            "For an analog scale, is the absolute reading error often taken as half the least count?",
            "Can half of the smallest division be used as a measurement uncertainty convention?",
            "True or false: least-count based uncertainty is commonly half a division for analog readings.",
        ])

    add("circuit_resistance", "current_same_in_series", "yes_no", "Yes",
        ["Elements in series form a single current path.", "The same current passes through every series element."],
        [
            "In a series circuit, is the current the same through all components?",
            "If two resistors are connected in series, do they carry the same current?",
            "True or false: series components share one common current.",
        ])
    add("circuit_resistance", "voltage_same_in_parallel", "yes_no", "Yes",
        ["Parallel branches connect across the same two nodes.", "Therefore each branch has the same voltage."],
        [
            "In a parallel circuit, is the voltage across each branch the same?",
            "Do parallel components share the same potential difference?",
            "True or false: branches in parallel have equal voltage.",
        ])
    add("circuit_resistance", "ammeter_connection", "text", "It is connected in series.",
        ["An ammeter measures the current through a component.", "It must be placed in series so the same current passes through it."],
        [
            "How should an ammeter be connected to measure current through a resistor?",
            "Should an ammeter be placed in series or parallel with the component being measured?",
            "To measure branch current, where is the ammeter placed?",
        ])
    add("circuit_resistance", "voltmeter_connection", "text", "It is connected in parallel.",
        ["A voltmeter measures potential difference between two points.", "It is connected in parallel across the component."],
        [
            "How should a voltmeter be connected to measure voltage across a resistor?",
            "Should a voltmeter be placed in series or parallel with the component?",
            "To measure potential difference, where is the voltmeter connected?",
        ])
    add("circuit_power", "higher_resistance_lower_current_fixed_voltage", "yes_no", "Yes",
        ["Ohm's law gives I = U/R.", "For fixed voltage, increasing resistance decreases current."],
        [
            "For a fixed voltage source, does increasing resistance reduce current?",
            "If U is unchanged and R becomes larger, should current become smaller?",
            "True or false: at constant voltage, current is inversely related to resistance.",
        ])
    add("circuit_power", "short_circuit_large_current", "yes_no", "Yes",
        ["A short circuit has very small resistance.", "For a given voltage, small resistance can produce a very large current."],
        [
            "Can a short circuit cause a very large current?",
            "Is low resistance the reason short circuits are dangerous?",
            "True or false: a near-zero resistance path can draw excessive current.",
        ])

    add("electrostatics_field", "equipotential_work_zero", "yes_no", "Yes",
        ["Moving a charge along an equipotential surface has no potential difference.", "Work by the electrostatic field for that displacement is zero."],
        [
            "Is the work zero when a charge moves along an equipotential surface?",
            "Along an equipotential line, is there any change in electric potential?",
            "True or false: no electrostatic work is needed along an equipotential path.",
        ])
    add("electrostatics_field", "field_perpendicular_equipotential", "yes_no", "Yes",
        ["Electric field points in the direction of greatest decrease of potential.", "It is perpendicular to equipotential surfaces."],
        [
            "Is the electric field perpendicular to equipotential surfaces?",
            "How is the electric field oriented relative to an equipotential surface?",
            "True or false: electric field lines cross equipotential surfaces at right angles.",
        ])
    add("electrostatics_field", "field_from_positive_charge_outward", "yes_no", "Yes",
        ["Electric field lines point away from positive charges.", "They point toward negative charges."],
        [
            "Do electric field lines point away from a positive point charge?",
            "Around a positive charge, is the electric field directed outward?",
            "True or false: the field of a positive charge points radially outward.",
        ])
    add("electrostatics_field", "field_from_negative_charge_inward", "yes_no", "Yes",
        ["Electric field lines point toward negative charges.", "This follows from the direction of force on a positive test charge."],
        [
            "Do electric field lines point toward a negative point charge?",
            "Around a negative charge, is the electric field directed inward?",
            "True or false: the field of a negative charge points toward the charge.",
        ])
    add("electrostatics_force", "newton_third_law_charges", "yes_no", "Yes",
        ["Two charges exert forces on each other with equal magnitude.", "The forces are opposite in direction, consistent with Newton's third law."],
        [
            "Do two point charges exert equal and opposite electric forces on each other?",
            "If charge A pushes charge B electrically, does B exert an equal opposite force on A?",
            "True or false: Coulomb forces between two charges form an action-reaction pair.",
        ])
    add("electrostatics_force", "uniform_field_force_direction", "yes_no", "Yes",
        ["In a uniform electric field, F = qE.", "A positive charge feels force along E, while a negative charge feels force opposite E."],
        [
            "In a uniform electric field, does a positive charge feel force in the direction of the field?",
            "Does a negative charge accelerate opposite to the electric field direction?",
            "True or false: the sign of charge controls the direction of force in a uniform field.",
        ])

    add("capacitor", "capacitor_blocks_dc_steady_state", "yes_no", "Yes",
        ["After a capacitor is fully charged in a DC circuit, no steady conduction current passes through the dielectric.", "The transient current occurs only while charging or discharging."],
        [
            "After a capacitor is fully charged by a DC source, does it block steady current?",
            "In steady DC conditions, does an ideal capacitor behave like an open circuit?",
            "True or false: a fully charged ideal capacitor passes no steady DC current.",
        ])
    add("capacitor", "capacitor_charge_increases_with_voltage", "yes_no", "Yes",
        ["For fixed capacitance, Q = C U.", "Increasing voltage increases stored charge."],
        [
            "For a fixed capacitor, does stored charge increase when voltage increases?",
            "If C is constant and U doubles, does Q also double?",
            "True or false: capacitor charge is proportional to voltage for fixed C.",
        ])
    add("capacitor", "capacitor_field_between_plates", "yes_no", "Yes",
        ["The electric field between ideal parallel plates is approximately uniform away from the edges.", "It is related to voltage by E = U/d."],
        [
            "Between large parallel capacitor plates, is the electric field approximately uniform?",
            "Can the field between ideal parallel plates be related to voltage by E = U/d?",
            "True or false: the parallel-plate capacitor field is nearly uniform in the central region.",
        ])

    add("induction", "flux_unit_weber", "text", "Weber.",
        ["Magnetic flux is measured in webers.", "One weber is equivalent to one tesla square metre."],
        [
            "What is the SI unit of magnetic flux?",
            "Which unit is used for magnetic flux in Faraday's law?",
            "Magnetic flux is reported in what SI unit?",
        ])
    add("induction", "emf_unit_volt", "text", "Volt.",
        ["Induced emf is an electric potential difference.", "Its SI unit is the volt."],
        [
            "What is the SI unit of induced emf?",
            "Induced electromotive force is measured in which unit?",
            "Which unit should be used for emf in Faraday's law?",
        ])
    add("induction", "moving_conductor_flux_change", "yes_no", "Yes",
        ["A conductor moving through a magnetic field can change the magnetic flux linked with a circuit.", "A changing flux can induce emf."],
        [
            "Can moving a conductor through a magnetic field induce an emf?",
            "If motion changes the flux linked with a circuit, can voltage be induced?",
            "True or false: electromagnetic induction can be caused by motion in a magnetic field.",
        ])
    add("induction", "inductor_opposes_current_change", "yes_no", "Yes",
        ["An inductor produces back emf when current changes.", "The induced effect opposes the change in current."],
        [
            "Does an inductor oppose changes in current?",
            "When current through an inductor changes, does back emf resist that change?",
            "True or false: an ideal inductor resists sudden current changes.",
        ])

    add("measurement_error", "absolute_error_same_unit", "yes_no", "Yes",
        ["Absolute error is a difference in the measured quantity.", "It has the same unit as the measured value."],
        [
            "Does absolute error have the same unit as the measured quantity?",
            "If length is measured in centimeters, is its absolute error also in centimeters?",
            "True or false: absolute uncertainty carries the original physical unit.",
        ])
    add("measurement_error", "relative_error_dimensionless", "yes_no", "Yes",
        ["Relative error is a ratio of absolute error to measured value.", "As a ratio of like units, it is dimensionless before being written as a percent."],
        [
            "Is relative error dimensionless before converting to percent?",
            "Does the unit cancel in a relative error ratio?",
            "True or false: relative uncertainty is a unitless ratio.",
        ])
    add("measurement_error", "smaller_relative_error_more_precise", "yes_no", "Yes",
        ["A smaller relative error means the uncertainty is smaller compared with the measured value.", "That indicates a more precise measurement in this context."],
        [
            "Does a smaller relative error indicate a more precise measurement?",
            "If two measurements have the same unit, is the one with lower relative error more precise?",
            "True or false: lower relative uncertainty usually means better precision.",
        ])

    add("LC_oscillation", "lc_period_formula_depends_lc", "yes_no", "Yes",
        ["The ideal LC period is T = 2*pi*sqrt(LC).", "It depends on inductance and capacitance."],
        [
            "Does the period of an ideal LC oscillator depend on L and C?",
            "Is T = 2*pi*sqrt(LC) the period relation for an ideal LC circuit?",
            "True or false: increasing L or C tends to increase the LC oscillation period.",
        ])
    add("LC_oscillation", "no_resistance_no_damping", "yes_no", "Yes",
        ["Resistance dissipates energy as heat.", "In the ideal LC model with no resistance, oscillations are not damped."],
        [
            "If an LC circuit has no resistance, are the ideal oscillations undamped?",
            "Does removing resistance eliminate energy loss in the ideal LC model?",
            "True or false: ideal LC oscillations continue without damping.",
        ])

    add("capacitor", "misconception_capacitance_voltage_dependent", "yes_no", "No",
        ["Ideal capacitance is set by geometry and dielectric.", "Changing voltage changes charge, not the capacitance itself."],
        [
            "True or false: doubling the voltage across an ideal capacitor doubles its capacitance.",
            "Does a capacitor's capacitance become larger simply because the applied voltage is larger?",
            "If the voltage source is stronger, must the capacitance of the same capacitor increase?",
        ])
    add("capacitor", "misconception_series_capacitors_add_directly", "yes_no", "No",
        ["Capacitors add directly only in parallel.", "For series capacitors, reciprocal capacitances add."],
        [
            "True or false: capacitors in series have equivalent capacitance C1 + C2.",
            "Do series capacitors add in the same direct way as parallel capacitors?",
            "Is the equivalent capacitance of two series capacitors always the sum of their capacitances?",
        ])
    add("induction", "misconception_static_flux_induces_emf", "yes_no", "No",
        ["Induced emf requires changing magnetic flux.", "A steady flux by itself does not induce emf."],
        [
            "True or false: any nonzero magnetic flux through a loop automatically creates induced emf.",
            "If the magnetic flux through a coil is steady, must there be an induced voltage?",
            "Can constant magnetic flux alone maintain an induced current in an ideal loop?",
        ])
    add("induction", "misconception_lenz_supports_change", "yes_no", "No",
        ["Lenz's law says the induced effect opposes the flux change.", "It does not reinforce the cause of the change."],
        [
            "True or false: Lenz's law says induced current helps the magnetic flux change that produced it.",
            "Does the induced current normally strengthen the change in flux that caused it?",
            "According to Lenz's law, does induction assist the original flux change?",
        ])
    add("LC_oscillation", "misconception_lc_energy_lost_ideal", "yes_no", "No",
        ["In the ideal LC model there is no resistance.", "Without dissipation, total electromagnetic energy is conserved."],
        [
            "True or false: an ideal LC oscillator loses energy on every cycle.",
            "Does an ideal LC circuit necessarily damp out even with zero resistance?",
            "In the no-resistance LC model, is total energy gradually destroyed?",
        ])
    add("LC_oscillation", "misconception_lc_frequency_amplitude", "yes_no", "No",
        ["The ideal LC frequency depends on L and C.", "It is independent of the initial voltage amplitude."],
        [
            "True or false: a larger initial voltage makes an ideal LC circuit oscillate at a higher frequency.",
            "Does increasing the starting charge change the natural frequency of an ideal LC oscillator?",
            "Is LC frequency controlled by amplitude rather than by L and C?",
        ])
    add("electrostatics_field", "misconception_field_scalar_addition", "yes_no", "No",
        ["Electric field has both magnitude and direction.", "Fields must be added vectorially, not as plain scalar magnitudes."],
        [
            "True or false: electric fields from several charges should always be added only by their magnitudes.",
            "Can net electric field be found by ignoring direction and adding all field sizes?",
            "Is electric-field superposition a scalar-only addition rule?",
        ])
    add("electrostatics_field", "misconception_positive_field_inward", "yes_no", "No",
        ["The field of a positive charge points outward.", "Field direction is the direction of force on a positive test charge."],
        [
            "True or false: electric field lines point inward toward a positive point charge.",
            "Around a positive charge, is the electric field directed into the charge?",
            "Do positive charges attract positive test charges inward as the field direction?",
        ])
    add("electrostatics_force", "misconception_like_charges_attract", "yes_no", "No",
        ["Like charges repel.", "Opposite charges attract."],
        [
            "True or false: two positive point charges attract each other.",
            "Do two charges with the same sign pull toward each other?",
            "If both charges are negative, is the Coulomb force attractive?",
        ])
    add("circuit_resistance", "misconception_parallel_resistance_increases", "yes_no", "No",
        ["Adding a parallel branch gives another path for current.", "The equivalent resistance decreases."],
        [
            "True or false: adding another resistor in parallel always increases equivalent resistance.",
            "Does total resistance become larger when a new parallel branch is added?",
            "For parallel resistors, is the equivalent resistance greater than every branch resistance?",
        ])
    add("circuit_resistance", "misconception_series_voltage_same", "yes_no", "No",
        ["Series elements have the same current.", "Their voltage drops generally divide according to resistance."],
        [
            "True or false: every resistor in a series circuit must have the same voltage drop.",
            "Do series resistors always share equal voltage regardless of their resistance?",
            "Is voltage necessarily identical across all series components?",
        ])
    add("circuit_power", "misconception_ideal_reactive_real_power", "yes_no", "No",
        ["Ideal inductors and capacitors store and return energy.", "Their average real power over a full AC cycle is zero."],
        [
            "True or false: an ideal inductor or capacitor consumes real power every AC cycle.",
            "Does a pure ideal reactive component dissipate average real power like a resistor?",
            "Can ideal reactance alone convert net electrical energy into heat?",
        ])
    add("measurement_error", "misconception_relative_error_has_unit", "yes_no", "No",
        ["Relative error is a ratio of like quantities.", "The unit cancels before it is reported as a percent."],
        [
            "True or false: relative error must have the same physical unit as the measured value.",
            "Does relative uncertainty keep units such as volts or meters?",
            "Is percent error a dimensional physical quantity?",
        ])
    add("measurement_error", "misconception_product_absolute_errors", "yes_no", "No",
        ["Products and quotients use relative-error propagation in the usual rule.", "Absolute errors are used directly for sums and differences."],
        [
            "True or false: for multiplication, absolute errors are added directly just like ordinary numbers.",
            "When calculating a product, should we ignore relative errors and add only absolute errors?",
            "For R = U/I, is the uncertainty found by simply adding volt and ampere errors as absolute quantities?",
        ])

    add("ac_resonance", "mcq_resonance_impedance", "mcq", "B. The resistance only",
        ["At resonance, inductive and capacitive reactances cancel.", "The series circuit impedance is therefore just R."],
        [
            "Choose the correct option for a series RLC circuit at resonance. A. Infinite impedance B. The resistance only C. Zero resistance D. Only capacitive reactance",
            "Which option describes the impedance of a series resonant RLC circuit? A. XL plus XC B. The resistance only C. Purely capacitive D. Purely inductive",
        ])
    add("capacitor", "mcq_parallel_capacitance", "mcq", "A. It increases",
        ["Parallel capacitances add directly.", "Adding another capacitor in parallel increases total capacitance."],
        [
            "What happens to equivalent capacitance when another capacitor is connected in parallel? A. It increases B. It becomes zero C. It must decrease D. It becomes negative",
            "Choose the correct statement about adding a capacitor in parallel. A. It increases equivalent capacitance B. It removes stored charge C. It lowers permittivity D. It reverses voltage",
        ])
    add("induction", "mcq_lenz_law", "mcq", "C. It opposes the flux change",
        ["Lenz's law gives the direction of induced current.", "The induced current opposes the change in magnetic flux."],
        [
            "What does Lenz's law state about induced current? A. It always helps the change B. It ignores magnetic flux C. It opposes the flux change D. It cancels resistance",
            "Select the correct Lenz-law statement. A. Current has no direction B. Flux never changes C. It opposes the flux change D. Voltage is always zero",
        ])

    return rows, log


def validate_row(row: dict[str, Any]) -> list[str]:
    errors = []
    messages = row.get("messages") or []
    if len(messages) < 3:
        return ["messages_too_short"]
    try:
        obj = json.loads(messages[-1]["content"])
    except Exception as exc:
        return [f"invalid_json:{exc}"]
    required = {"question_kind", "topic", "concept", "answer_type", "answer", "evidence", "confidence"}
    missing = required - set(obj)
    if missing:
        errors.append("missing:" + ",".join(sorted(missing)))
    if "python_code" in obj:
        errors.append("forbidden_python_code")
    if obj.get("question_kind") != "conceptual":
        errors.append("not_conceptual")
    if not isinstance(obj.get("evidence"), list) or not obj.get("evidence"):
        errors.append("bad_evidence")
    if obj.get("answer_type") == "yes_no" and obj.get("answer") not in {"Yes", "No", "Uncertain"}:
        errors.append("bad_yes_no_answer")
    if not isinstance(obj.get("confidence"), (int, float)):
        errors.append("bad_confidence")
    return errors


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verified = build_verified_rows()
    seeds = build_seed_rows()
    synthetic, synth_log = build_synthetic_rows()
    concept_bank, concept_bank_log = build_curated_concept_bank_rows()
    rows_by_id: dict[str, dict[str, Any]] = {}
    for item in [*verified, *seeds, *synthetic, *concept_bank]:
        rows_by_id[item["id"]] = item
    rows = list(rows_by_id.values())

    rejected = []
    kept = []
    for item in rows:
        errors = validate_row(item)
        if errors:
            rejected.append({"id": item.get("id"), "errors": ";".join(errors)})
        else:
            kept.append(item)

    random.Random(20260614).shuffle(kept)
    valid_size = max(24, int(len(kept) * 0.18))
    valid = kept[:valid_size]
    train = kept[valid_size:]
    write_jsonl(OUT_DIR / "chlt_reasoner_final_all.jsonl", kept)
    write_jsonl(OUT_DIR / "chlt_reasoner_final_train.jsonl", train)
    write_jsonl(OUT_DIR / "chlt_reasoner_final_valid.jsonl", valid)
    (OUT_DIR / "chlt_reasoner_final_synthetic_log.json").write_text(json.dumps(synth_log, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "chlt_reasoner_final_concept_bank_log.json").write_text(json.dumps(concept_bank_log, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "chlt_reasoner_final_rejected.json").write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")

    source_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}
    answer_counts: dict[str, int] = {}
    for item in kept:
        meta = item["metadata"]
        source_counts[meta["source"]] = source_counts.get(meta["source"], 0) + 1
        topic_counts[meta["topic"]] = topic_counts.get(meta["topic"], 0) + 1
        answer_counts[meta["answer"]] = answer_counts.get(meta["answer"], 0) + 1
    summary = {
        "total_rows": len(kept),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "verified_rows": len(verified),
        "seed_rows": len(seeds),
        "synthetic_rows": len(synthetic),
        "concept_bank_rows": len(concept_bank),
        "rejected_rows": len(rejected),
        "source_counts": source_counts,
        "topic_counts": topic_counts,
        "answer_counts": answer_counts,
        "policy": "CHLT final uses verified CHLT source rows, seed conceptual rows, formula-verified synthetic RLC rows, and curated concept-bank augmentation. Output is JSON-only conceptual answer with evidence; no Python code.",
    }
    (OUT_DIR / "chlt_reasoner_final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
