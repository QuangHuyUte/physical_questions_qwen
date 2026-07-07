# Calculator Coverage Audit

Date: 2026-06-14

## Summary

The calculator is internally consistent: every formula in `src/formula_bank.py`
has a corresponding implementation branch in `src/calculator.py`.

- Formula bank entries after expansion: 69
- Implemented calculator branches after expansion: 69
- Missing implementation for registered formula IDs: 0
- Implemented but unregistered formula IDs: 0

Update after expansion:

- Added missing `Q-C-U-W` capacitor rearrangements.
- Added inductor current from stored energy and inductance.
- Added wire resistance from resistivity, length, and cross-sectional area.
- Added dielectric and voltage-distance parallel-plate relations.
- Added two-vector resultant and common equal-charge geometry shortcuts.
- Added capacitive reactance, series RLC current, and series RLC power.
- Added power rearrangements from `P,R`.
- Added basic absolute-error helpers.
- Added `src/payload_validator.py` to detect forbidden output keys,
  unknown formula IDs, unreadable givens, and missing formula roles before a
  parser payload is trusted.

However, the calculator is not yet complete for all likely EXACT Type 2
physics variants. The rejected dataset rows show several recurring families that
are outside the current formula bank or require geometry/state reasoning beyond
the simple scalar formulas.

## Current Covered Families

The current formula bank covers these groups reasonably well:

- Ohm law: `R = U/I`, `I = U/R`, `U = IR`
- Electric power: `P = UI`, `P = I^2R`, `P = U^2/R`, `P = A/t`
- Series/parallel resistors and capacitors
- Basic capacitor relations: `Q = CU`, `W = 1/2 CU^2`, and rearrangements from
  energy/voltage/capacitance
- Basic inductor energy: `W = 1/2 LI^2` and rearrangement for `L`
- Ideal LC: frequency, angular frequency, peak current from energy conservation
- Basic RLC resonance: find `C` from `L,f`, find `L` from `C,f`
- Simple Coulomb force between two point charges
- Simple point-charge electric field
- Force in a uniform electric field: `F = qE`
- Basic magnetic flux and Faraday/self-induction formulas
- Long-solenoid magnetic field
- Basic measurement relative error and resistance uncertainty

## Main Missing / Weak Families

These are the main gaps found from `numeric_parser_rejected.csv`.

### 1. Electrostatic Vector Geometry

Examples in rejected rows:

- Equilateral triangle field/force at a vertex or centroid
- Isosceles right triangle net force at the right-angle vertex
- Two charges plus test charge on perpendicular bisector
- Field at midpoint or off-axis point from two charges
- Resultant field when two component fields form a known angle

Needed formula family:

- `vector_resultant_two_components`
- `electric_field_two_charges_line`
- `electric_field_two_charges_perpendicular_bisector`
- `electric_field_equilateral_triangle_vertex`
- `coulomb_force_two_sources_on_test_charge`
- `coulomb_force_right_triangle_equal_charges`

### 2. General Vector Resultant

Examples:

- Two forces with magnitudes `F1`, `F2` and angle `theta`
- Special cases: 90°, 120°, 135°

Needed formula:

- `resultant_two_vectors_from_angle`:  
  `R = sqrt(F1^2 + F2^2 + 2 F1 F2 cos(theta))`

### 3. AC/RLC General Impedance and Frequency Scaling

Current support exists, but rejected rows show extraction/logic gaps.

Examples:

- `R, XL, XC` not at resonance, ask total impedance
- Frequency multiplied by `k`, ask RMS current
- Frequency multiplied by `k`, ask voltage across resistor
- Phase/segment problems: `u_AM` out of phase with `u_MB`

Needed improvement:

- Strengthen parser for `XL`, `XC`, `R`, `U`, `frequency_factor`
- Add robust formula families:
  - `rlc_series_impedance_from_R_XL_XC`
  - `rlc_scaled_current_from_XL_XC_R_U_k`
  - `rlc_scaled_resistor_voltage_general`
  - possibly leave phase/segment problems as advanced/unsupported unless enough
    clean training data exists

### 4. Measurement Error Generalization

Current measurement support is too narrow.

Examples:

- Least count implies absolute error or half least count depending convention
- `x = measured ± delta`, ask percentage relative error
- Actual vs measured: absolute and relative error
- Multiple measurements: mean and mean absolute error
- Product/quotient error: `P = UI`, `R = U/I`
- Series resistance uncertainty

Needed formula families:

- `absolute_error_from_least_count`
- `relative_error_from_measured_plus_minus`
- `absolute_and_relative_error_from_true_measured`
- `mean_value_and_mean_absolute_error`
- `relative_error_product_or_quotient`
- `absolute_error_product_or_quotient`
- `absolute_error_sum`

### 5. Capacitor State Transformations

Current support covers direct `Q, C, U, W`, but not state changes.

Examples:

- Isolated capacitor, plate distance changes
- Connected to source, plate distance changes
- Charge sharing among identical capacitors
- Disconnected capacitor connected to another uncharged capacitor
- Voltage doubled/halved, ask energy factor
- Dielectric inserted/replaced, ask capacitance or energy change
- Short-circuit: final charge/energy zero

Needed formula families:

- `capacitor_energy_isolated_after_capacitance_change`
- `capacitor_energy_source_connected_after_capacitance_change`
- `capacitor_charge_sharing_identical_capacitors`
- `capacitor_energy_scaling_with_voltage`
- `capacitor_energy_scaling_with_capacitance`
- `parallel_plate_capacitance_with_dielectric`
- `capacitor_force_between_plates`
- `capacitor_energy_density`

### 6. Missing Rearrangements of Existing Relations

Several simple rearrangements are absent or underrepresented:

- `C = Q/U`
- `U = Q/C`
- `W = QU/2`
- `C = eps_r eps0 A/d`
- `Q = eps0 eps_r A U / d`
- `I = sqrt(2W/L)` from inductor energy
- `U_max` or `C/L` variants in LC energy

### 7. Charged Plane / Parallel Plate Electric Field

Examples:

- Infinite charged plate with surface charge density or total charge over area
- Equilibrium of charged particle between plates: `qE = mg`

Needed formula families:

- `electric_field_infinite_plane_from_surface_charge_density`
- `electric_field_between_parallel_plates_from_force_balance`
- `electric_field_parallel_plate_from_voltage_distance`

### 8. Time-Dependent Sinusoidal Quantities

Examples:

- `I(t) = I0 sin(omega t)` ask maximum magnetic energy
- `U(t) = U0 sin(omega t)` ask maximum capacitor energy

Needed parser handling:

- Extract amplitude from sinusoidal expression.

Potential formulas can reuse:

- `inductor_energy_from_L_I`
- `capacitor_energy_from_C_U`

### 9. Wire Resistivity

Examples:

- `rho`, length, area, ask resistance.

Needed formula:

- `wire_resistance_from_resistivity_length_area`: `R = rho*l/S`

### 10. Multi-Output and Theory-Like Numeric Rows

Some rows ask multiple outputs:

- Calculate energy and charge
- Calculate current through each bulb and total current
- Mean and mean absolute error

The current API Type 2 response seems to expect one answer object, so these
should be either:

- routed to a multi-output handler, or
- excluded from numeric single-answer calculator paths unless the expected
  answer format is standardized.

## Low-Representation Existing Formula IDs

These formulas exist but have fewer than 10 examples in the current parser
dataset, so numeric parser accuracy may be weak:

- `angular_frequency_from_frequency`: 1
- `capacitance_series`: 1
- `electric_field_from_force_charge`: 1
- `magnetic_flux_from_solenoid_area`: 1
- `power_total_sum`: 1
- `resistance_uncertainty_from_voltage_current`: 1
- `current_from_power_voltage`: 2
- `series_ac_impedance_from_R_XL_XC`: 2
- `solenoid_field_from_turns_length_current`: 2
- `faraday_emf_from_flux_change`: 3
- `inductance_from_emf_current_change`: 3
- `inductive_reactance_from_L_f`: 3
- `magnetic_flux_from_B_area_angle`: 4
- `power_from_energy_time`: 5
- `self_induction_emf_from_L_current_change`: 6
- `voltage_from_current_resistance`: 6
- `resistance_parallel`: 7
- `power_from_voltage_current`: 8

## Recommendation

The next calculator expansion should not add one-off question patterns. It
should add formula families in this order:

1. Measurement error generalization beyond the common single-value cases
2. Electrostatic vector geometry beyond simple angle/equal-charge shortcuts
3. Capacitor state transformations such as source-connected vs isolated changes
4. Time-dependent sinusoidal amplitude extraction
5. Multi-output numeric rows

After each family is added:

1. Add formula spec to `formula_bank.py`
2. Implement branch in `calculator.py`
3. Add or regenerate verified parser examples
4. Run 100-case stress test
5. Keep unsupported cases in a log rather than letting the LLM solve them
