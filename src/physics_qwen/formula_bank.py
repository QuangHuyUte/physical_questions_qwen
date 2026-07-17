from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FormulaSpec:
    formula_id: str
    target_roles: tuple[str, ...]
    required_roles: tuple[str, ...]
    expression: str
    notes: str


FORMULA_BANK: tuple[FormulaSpec, ...] = (
    FormulaSpec("resistance_from_voltage_current", ("resistance",), ("voltage", "current"), "R = U/I", "Ohm law."),
    FormulaSpec("resistance_from_resonance_impedance", ("resistance",), ("impedance",), "At resonance, R = Z", "Series RLC at resonance."),
    FormulaSpec("current_from_voltage_resistance", ("current",), ("voltage", "resistance"), "I = U/R", "Ohm law."),
    FormulaSpec("voltage_from_current_resistance", ("voltage",), ("current", "resistance"), "U = I*R", "Ohm law."),
    FormulaSpec("power_from_voltage_current", ("power",), ("voltage", "current"), "P = U*I", "DC or RMS AC power relation."),
    FormulaSpec("power_from_current_resistance", ("power",), ("current", "resistance"), "P = I^2*R", "Joule heating."),
    FormulaSpec("power_from_voltage_resistance", ("power",), ("voltage", "resistance"), "P = U^2/R", "Joule heating."),
    FormulaSpec("power_from_energy_time", ("power",), ("energy", "time"), "P = A/t", "Average power from work/energy over time."),
    FormulaSpec("power_total_sum", ("power",), ("power",), "P_total = sum(P_i)", "Total power is the sum of component powers."),
    FormulaSpec("current_from_power_voltage", ("current",), ("power", "voltage"), "I = P/U", "Power relation rearranged."),
    FormulaSpec("voltage_from_power_current", ("voltage",), ("power", "current"), "U = P/I", "Power relation rearranged."),
    FormulaSpec("current_from_power_resistance", ("current",), ("power", "resistance"), "I = sqrt(P/R)", "Joule heating rearranged."),
    FormulaSpec("voltage_from_power_resistance", ("voltage",), ("power", "resistance"), "U = sqrt(P*R)", "Joule heating rearranged."),
    FormulaSpec("wire_resistance_from_resistivity_length_area", ("resistance",), ("resistivity", "distance", "area"), "R = rho*l/S", "Uniform wire resistance."),
    FormulaSpec("resistance_series", ("resistance",), ("resistance",), "R_eq = sum(R_i)", "Series connection."),
    FormulaSpec("resistance_parallel", ("resistance",), ("resistance",), "1/R_eq = sum(1/R_i)", "Parallel connection."),
    FormulaSpec("total_current_parallel_resistors_from_voltage", ("current",), ("voltage", "resistance"), "I_total = U*sum(1/R_i)", "Parallel branches share the same voltage."),
    FormulaSpec("capacitance_series", ("capacitance",), ("capacitance",), "1/C_eq = sum(1/C_i)", "Series capacitors."),
    FormulaSpec("capacitance_parallel", ("capacitance",), ("capacitance",), "C_eq = sum(C_i)", "Parallel capacitors."),
    FormulaSpec("capacitor_charge_from_C_U", ("charge",), ("capacitance", "voltage"), "Q = C*U", "Capacitor definition."),
    FormulaSpec("capacitor_capacitance_from_charge_voltage", ("capacitance",), ("charge", "voltage"), "C = Q/U", "Capacitor definition rearranged."),
    FormulaSpec("capacitor_voltage_from_charge_capacitance", ("voltage",), ("charge", "capacitance"), "U = Q/C", "Capacitor definition rearranged."),
    FormulaSpec("capacitor_energy_from_C_U", ("energy",), ("capacitance", "voltage"), "W = 0.5*C*U^2", "Energy in capacitor."),
    FormulaSpec("capacitor_energy_from_Q_U", ("energy",), ("charge", "voltage"), "W = 0.5*Q*U", "Capacitor energy relation."),
    FormulaSpec("capacitor_energy_from_Q_C", ("energy",), ("charge", "capacitance"), "W = Q^2/(2C)", "Capacitor energy relation."),
    FormulaSpec("capacitor_capacitance_from_energy_voltage", ("capacitance",), ("energy", "voltage"), "C = 2W/U^2", "Energy relation rearranged."),
    FormulaSpec("capacitor_voltage_from_energy_capacitance", ("voltage",), ("energy", "capacitance"), "U = sqrt(2W/C)", "Energy relation rearranged."),
    FormulaSpec("parallel_plate_capacitance_with_dielectric", ("capacitance",), ("dielectric_constant", "area", "distance"), "C = eps_r*eps0*A/d", "Parallel-plate capacitor with dielectric."),
    FormulaSpec("parallel_plate_field_from_voltage_distance", ("electric_field",), ("voltage", "distance"), "E = U/d", "Uniform field between plates."),
    FormulaSpec("inductor_energy_from_L_I", ("energy",), ("inductance", "current"), "W = 0.5*L*I^2", "Energy in inductor."),
    FormulaSpec("inductance_from_energy_current", ("inductance",), ("energy", "current"), "L = 2W/I^2", "Inductor energy rearranged."),
    FormulaSpec("inductor_current_from_energy_inductance", ("current",), ("energy", "inductance"), "I = sqrt(2W/L)", "Inductor energy rearranged."),
    FormulaSpec("lc_max_current_from_voltage_capacitance_inductance", ("current",), ("voltage", "capacitance", "inductance"), "I0 = U0*sqrt(C/L)", "LC energy conservation."),
    FormulaSpec("lc_angular_frequency_from_L_C", ("angular_frequency",), ("inductance", "capacitance"), "omega = 1/sqrt(LC)", "Ideal LC oscillator."),
    FormulaSpec("lc_frequency_from_L_C", ("frequency",), ("inductance", "capacitance"), "f = 1/(2*pi*sqrt(LC))", "Ideal LC oscillator."),
    FormulaSpec("rlc_resonance_capacitance_from_L_f", ("capacitance",), ("inductance", "frequency"), "C = 1/((2*pi*f)^2*L)", "RLC resonance."),
    FormulaSpec("rlc_resonance_inductance_from_C_f", ("inductance",), ("capacitance", "frequency"), "L = 1/((2*pi*f)^2*C)", "RLC resonance."),
    FormulaSpec("parallel_plate_capacitance_from_area_distance", ("capacitance",), ("area", "distance"), "C = eps0*A/d", "Ideal air-filled parallel-plate capacitor."),
    FormulaSpec("coulomb_force_two_charges", ("force",), ("charge", "charge", "distance"), "F = k*|q1*q2|/r^2", "Point-charge force."),
    FormulaSpec("coulomb_force_right_angle_equal_charges", ("force",), ("charge", "distance"), "F_net = sqrt(2)*k*q^2/r^2", "Net force from two equal perpendicular Coulomb forces."),
    FormulaSpec("coulomb_force_equilateral_equal_charges", ("force",), ("charge", "distance"), "F_net = sqrt(3)*k*q^2/r^2", "Net force at a vertex of an equilateral triangle of equal charges."),
    FormulaSpec("force_from_charge_field", ("force",), ("charge", "electric_field"), "F = |q|E", "Uniform electric field."),
    FormulaSpec("electric_field_from_force_charge", ("electric_field",), ("force", "charge"), "E = F/|q|", "Uniform electric field."),
    FormulaSpec("electric_field_point_charge_or_superposition", ("electric_field",), ("charge", "distance"), "E = k*|q|/r^2", "Single point-charge field."),
    FormulaSpec("electric_field_two_charges_angle", ("electric_field",), ("charge", "charge", "distance"), "E = sqrt(E1^2+E2^2+2E1E2*cos(theta))", "Vector sum of two point-charge fields at a known angle."),
    FormulaSpec("resultant_two_vectors_from_angle", ("force",), ("force", "force"), "R = sqrt(A^2+B^2+2AB*cos(theta))", "Resultant of two force vectors with a known angle."),
    FormulaSpec("electric_field_resultant_two_vectors_from_angle", ("electric_field",), ("electric_field", "electric_field"), "E = sqrt(E1^2+E2^2+2E1E2*cos(theta))", "Resultant of two electric-field vectors with a known angle."),
    FormulaSpec("magnetic_flux_from_B_area_angle", ("magnetic_flux",), ("magnetic_field", "area"), "Phi = B*S*cos(theta)", "Magnetic flux."),
    FormulaSpec("magnetic_flux_linkage_from_turns_flux", ("magnetic_flux",), ("turns", "magnetic_flux"), "lambda = N*Phi", "Flux linkage."),
    FormulaSpec("magnetic_flux_from_solenoid_area", ("magnetic_flux",), ("turn_density", "current", "area"), "Phi = mu0*n*I*S", "Flux through one turn inside a long solenoid."),
    FormulaSpec("faraday_emf_from_flux_change", ("voltage",), ("magnetic_flux", "time"), "|e| = |Delta Phi|/Delta t", "Faraday law for one turn."),
    FormulaSpec("self_induction_emf_from_L_current_change", ("voltage",), ("inductance", "current", "time"), "|e| = L*|Delta I|/Delta t", "Self-induced emf."),
    FormulaSpec("inductance_from_emf_current_change", ("inductance",), ("voltage", "current", "time"), "L = |e|*Delta t/|Delta I|", "Self-induction rearranged."),
    FormulaSpec("solenoid_field_from_turn_density_current", ("magnetic_field",), ("turn_density", "current"), "B = mu0*n*I", "Long solenoid field."),
    FormulaSpec("solenoid_field_from_turns_length_current", ("magnetic_field",), ("turns", "distance", "current"), "B = mu0*(N/l)*I", "Long solenoid field using total turns and length."),
    FormulaSpec("series_ac_impedance_from_R_XL_XC", ("resistance",), ("resistance", "inductive_reactance", "capacitive_reactance"), "Z = sqrt(R^2+(XL-XC)^2)", "Series RLC impedance."),
    FormulaSpec("inductive_reactance_from_L_f", ("resistance",), ("inductance", "frequency"), "X_L = 2*pi*f*L", "Inductive reactance."),
    FormulaSpec("capacitive_reactance_from_C_f", ("resistance",), ("capacitance", "frequency"), "X_C = 1/(2*pi*f*C)", "Capacitive reactance."),
    FormulaSpec("rlc_series_current_from_U_R_XL_XC", ("current",), ("voltage", "resistance", "inductive_reactance", "capacitive_reactance"), "I = U/Z", "Series RLC current from impedance."),
    FormulaSpec("rlc_series_power_from_U_R_XL_XC", ("power",), ("voltage", "resistance", "inductive_reactance", "capacitive_reactance"), "P = U^2*R/Z^2", "Average power in a series RLC circuit."),
    FormulaSpec("rlc_frequency_scaled_current", ("current",), ("resistance", "inductive_reactance", "capacitive_reactance", "frequency_factor", "voltage"), "I = U/sqrt(R^2+(kXL-XC/k)^2)", "Series RLC after frequency scaling."),
    FormulaSpec("rlc_frequency_scaled_resistor_voltage", ("voltage",), ("inductive_reactance", "capacitive_reactance", "frequency_factor", "voltage"), "If kXL = XC/k, then U_R = U", "Resistor voltage after frequency scaling to resonance."),
    FormulaSpec("angular_frequency_from_frequency", ("angular_frequency",), ("frequency",), "omega = 2*pi*f", "Angular frequency."),
    FormulaSpec("relative_error_from_absolute", ("relative_error",), ("absolute_error",), "relative error = Delta x/x*100%", "Measurement relative error."),
    FormulaSpec("absolute_error_from_relative", ("absolute_error",), ("relative_error", "measured_value"), "Delta x = relative_error*x/100", "Absolute error from relative error."),
    FormulaSpec("absolute_error_from_least_count_half", ("absolute_error",), ("least_count",), "Delta x = least_count/2", "Instrument uncertainty convention."),
    FormulaSpec("relative_error_product_or_quotient", ("relative_error",), ("voltage", "current"), "relative error = sum(Delta x/x)*100%", "Relative errors add for products and quotients."),
    FormulaSpec("resistance_uncertainty_from_voltage_current", ("resistance_uncertainty",), ("voltage", "current"), "Delta R = R(Delta U/U + Delta I/I)", "Maximum uncertainty propagation."),
    FormulaSpec("absolute_error_difference", ("absolute_error",), ("measured_value", "actual_value"), "Delta x = |x_measured - x_actual|", "Absolute error from actual and measured values."),
)


FORMULA_BY_ID = {spec.formula_id: spec for spec in FORMULA_BANK}


def candidate_ids(payload: dict) -> list[str]:
    ids: list[str] = []
    for item in payload.get("formula_candidates") or []:
        if isinstance(item, dict) and item.get("formula_id"):
            ids.append(str(item["formula_id"]))
    return ids


def ordered_formula_ids(payload: dict) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for formula_id in [*candidate_ids(payload), *(spec.formula_id for spec in FORMULA_BANK)]:
        if formula_id not in seen:
            ordered.append(formula_id)
            seen.add(formula_id)
    return ordered


def describe_formula(formula_id: str) -> str:
    spec = FORMULA_BY_ID.get(formula_id)
    return spec.expression if spec else formula_id


def formula_known(formula_id: str) -> bool:
    return formula_id in FORMULA_BY_ID
