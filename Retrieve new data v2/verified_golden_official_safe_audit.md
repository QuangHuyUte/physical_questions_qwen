# Verified Golden Official-Safe Audit

## Files Created
- `Retrieve new data v2\verified_golden_official_safe.csv`

## Policy
- The original `cot` column from `verified_golden_expanded.csv` is not copied into the safe file.
- For IDs present in the official physics dataset, `official_cot` is copied from `Physics_Problems_Text_Only_v2.csv`.
- Augmented rows are kept, but `official_cot` is blank and `cot_source` marks them as synthetic/disclosable.
- `topic`, `prefix`, and `golden_code` are kept as derived/internal metadata, not official annotations.

## Summary
- Verified rows: 1660
- Official rows: 1755
- Rows overlapping official by id: 1149
- Augmented rows kept: 511
- Question differences kept as verified corrections: 22
- Verified CoT differences replaced by official CoT: 515
- Answer differences kept as audited verified corrections: 10
- Unit differences kept as verified normalizations/corrections: 60
- SFT official-CoT seed rows: 1149

## Answer Differences Kept From Verified
| id | prefix | topic | official_answer | official_unit | verified_answer | verified_unit | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CH212 | CH | ac_resonance | 707 | - | 0.707 | - | kept verified answer as audited correction |
| CH218 | CH | capacitor | 142.35 | W | 142.35294117647058 | W | kept verified answer as audited correction |
| CH243 | CH | circuit_power | 171.43 | W | 192 | W | kept verified answer as audited correction |
| DT098 | DT | electrostatics_field | 4.0 × 10⁴ | N/C | 3.2 x 10^5 | N/C | kept verified answer as audited correction |
| LD303 | LD | electrostatics_field | 7.95 × 10^6 | V/m | 5.65 × 10^6 | V/m | kept verified answer as audited correction |
| LD395 | LD | electrostatics_field | 7.42*10^6 | V/m | 10.01 × 10^6 | V/m | kept verified answer as audited correction |
| TD179 | TD | capacitor | 283.2 | nJ | 283.1 | nJ | kept verified answer as audited correction |
| TD181 | TD | capacitor | 1.46 | nC | 1.45 | nC | kept verified answer as audited correction |
| TD369 | TD | capacitor | Do not change |  | 100 | μC | kept verified answer as audited correction |
| TD401 | TD | capacitor | 45 | J | 0.045 | J | kept verified answer as audited correction |

## Unit Differences Kept From Verified
| id | prefix | topic | official_unit | verified_unit | decision |
| --- | --- | --- | --- | --- | --- |
| CH349 | CH | capacitor | µF | μF | kept verified unit normalization/correction |
| CH350 | CH | general_physics | µF | μF | kept verified unit normalization/correction |
| CH351 | CH | capacitor | µF | μF | kept verified unit normalization/correction |
| CH352 | CH | general_physics | µF | μF | kept verified unit normalization/correction |
| CH353 | CH | general_physics | µF | μF | kept verified unit normalization/correction |
| CH354 | CH | capacitor | µF | μF | kept verified unit normalization/correction |
| CH355 | CH | general_physics | µF | μF | kept verified unit normalization/correction |
| CH356 | CH | capacitor | µF | μF | kept verified unit normalization/correction |
| CH373 | CH | general_physics |  | - | kept verified unit normalization/correction |
| CH374 | CH | general_physics |  | - | kept verified unit normalization/correction |
| DDT140 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT143 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT145 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT146 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT149 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT152 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT153 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT156 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT206 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT207 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT210 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT217 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT219 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT220 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT327 | DDT | ac_resonance |  | - | kept verified unit normalization/correction |
| DDT330 | DDT | LC_oscillation | — | - | kept verified unit normalization/correction |
| DDT337 | DDT | ac_resonance |  | - | kept verified unit normalization/correction |
| DDT340 | DDT | general_physics | — | - | kept verified unit normalization/correction |
| DDT343 | DDT | ac_resonance |  | - | kept verified unit normalization/correction |
| DDT350 | DDT | LC_oscillation | — | - | kept verified unit normalization/correction |
| DDT352 | DDT | LC_oscillation | — | - | kept verified unit normalization/correction |
| DDT354 | DDT | induction | — | - | kept verified unit normalization/correction |
| DDT360 | DDT | induction | — | - | kept verified unit normalization/correction |
| DT047 | DT | electrostatics_field |  | - | kept verified unit normalization/correction |
| LD020 | LD | electrostatics_force | Độ | degree | kept verified unit normalization/correction |
| NL025 | NL | LC_oscillation | — | - | kept verified unit normalization/correction |
| NL026 | NL | LC_oscillation | — | - | kept verified unit normalization/correction |
| NL100 | NL | LC_oscillation | — | - | kept verified unit normalization/correction |
| NL119 | NL | LC_oscillation | — | - | kept verified unit normalization/correction |
| NL120 | NL | LC_oscillation | — | - | kept verified unit normalization/correction |
| NL127 | NL | capacitor | lần | times | kept verified unit normalization/correction |
| NL319 | NL | capacitor | µF | μF | kept verified unit normalization/correction |
| NL333 | NL | capacitor | µF | μF | kept verified unit normalization/correction |
| NL353 | NL | capacitor | µF | μF | kept verified unit normalization/correction |
| NL370 | NL | capacitor | µF | μF | kept verified unit normalization/correction |
| NL395 | NL | capacitor | µF | μF | kept verified unit normalization/correction |
| TD093 | TD | capacitor | — | - | kept verified unit normalization/correction |
| TD094 | TD | capacitor | — | - | kept verified unit normalization/correction |
| TD098 | TD | capacitor | — | - | kept verified unit normalization/correction |
| TD100 | TD | capacitor | — | - | kept verified unit normalization/correction |
| TD367 | TD | capacitor | lần | times | kept verified unit normalization/correction |
| TD369 | TD | capacitor |  | μC | kept verified unit normalization/correction |
| TD371 | TD | capacitor | lần | times | kept verified unit normalization/correction |
| TD377 | TD | capacitor |  | - | kept verified unit normalization/correction |
| TD380 | TD | capacitor |  | - | kept verified unit normalization/correction |
| TD386 | TD | capacitor |  | - | kept verified unit normalization/correction |
| THCB071 | THCB | circuit_resistance | — | - | kept verified unit normalization/correction |
| THCB073 | THCB | circuit_resistance | — | - | kept verified unit normalization/correction |
| THCB081 | THCB | circuit_resistance | — | - | kept verified unit normalization/correction |
| THCB083 | THCB | circuit_resistance | — | - | kept verified unit normalization/correction |

## Question Differences Kept From Verified
| id | prefix | topic | official_question | verified_question | decision |
| --- | --- | --- | --- | --- | --- |
| CH190 | CH | ac_resonance | Consider a series RLC circuit. At the initial angular frequency ω0, XL = 45 Ω and XC = 405 Ω. The new angular frequency is set to k·ω0. What is the value of k for the circuit to... | Consider a series RLC circuit. At the initial angular frequency ω0, XL = 45 Ω and XC = 405 Ω. The new angular frequency is set to k×ω0. What is the value of k for the circuit to... | kept verified question correction |
| CH195 | CH | ac_resonance | Consider a series RLC circuit. At the initial angular frequency ω0, X_L = 80 Ω and X_C = 20 Ω. A new angular frequency must be set to k·ω0. What is the value of k for the circui... | Consider a series RLC circuit. At the initial angular frequency ω0, X_L = 80 Ω and X_C = 20 Ω. A new angular frequency must be set to k×ω0. What is the value of k for the circui... | kept verified question correction |
| CH200 | CH | ac_resonance | Consider a series RLC circuit. At an initial angular frequency ω₀, the inductive reactance X_L is 16 Ω, and the capacitive reactance X_C is 32 Ω. What is the value of k for the ... | Consider a series RLC circuit. At an initial angular frequency ω₀, the inductive reactance X_L is 16 Ω, and the capacitive reactance X_C is 32 Ω. What is the value of k for the ... | kept verified question correction |
| CH205 | CH | ac_resonance | Consider an RLC series circuit. At the initial angular frequency ω0, X_L = 48 Ω and X_C = 192 Ω. The new angular frequency must be set to k·ω0. What is the value of k for the ci... | Consider an RLC series circuit. At the initial angular frequency ω0, X_L = 48 Ω and X_C = 192 Ω. The new angular frequency must be set to k×ω0. What is the value of k for the ci... | kept verified question correction |
| CH210 | CH | ac_resonance | Consider a series RLC circuit. At an initial angular frequency ω0, X_L = 95 Ω and X_C = 380 Ω. The new angular frequency must be set to k·ω0. What is the value of k for the circ... | Consider a series RLC circuit. At an initial angular frequency ω0, X_L = 95 Ω and X_C = 380 Ω. The new angular frequency must be set to k×ω0. What is the value of k for the circ... | kept verified question correction |
| CH215 | CH | ac_resonance | Consider a series RLC circuit. At the initial angular frequency ω₀, X_L = 20 Ω and X_C = 180 Ω. The new angular frequency is set to k·ω₀. What is the value of k for the circuit ... | Consider a series RLC circuit. At the initial angular frequency ω₀, X_L = 20 Ω and X_C = 180 Ω. The new angular frequency is set to k×ω₀. What is the value of k for the circuit ... | kept verified question correction |
| CH377 | CH | ac_resonance | Here are a few ways to translate that question, depending on desired conciseness and phrasing: **Option 1 (Direct and common for physics problems):** "At resonance, with U = 200... | At resonance, with U = 200 V and R = 25 Ω, calculate the power P. | kept verified question correction |
| DT008 | DT | electrostatics_field | Two charges, q1 = q2 = q (where q > 0, in Coulombs), are placed at points A and B, with the distance AB = 2a (meters). Point M is located on the perpendicular bisector of the li... | Two charges, q1 = q2 = q (where q > 0, in Coulombs), are placed at points A and B, with the distance AB = 2a (meters). Point M is located on the perpendicular bisector of the li... | kept verified question correction |
| LD002 | LD | electrostatics_force | Three electric charges are placed at three fixed points, forming a right-angled triangle ABC, where AB = 4 m and BC = 5 m. The charges are qA = 5.0 μC, qB = -5.0 μC, and qC = 4.... | Three electric charges are placed at three fixed points, forming a right-angled triangle ABC (right-angled at A), where AB = 4 m and BC = 5 m. The charges are qA = 5.0 μC, qB = ... | kept verified question correction |
| LD021 | LD | electrostatics_force | A charge q = -1 μC is attracted by two +1 μC charges. These two positive charges are located on opposite sides of q, along the same straight line passing through q, at distances... | A charge q = -1 μC is attracted by two +1 μC charges. These two positive charges are located on opposite sides of q, along the same straight line passing through q, at distances... | kept verified question correction |
| LD040 | LD | electrostatics_force | Two point charges q1 = 3 μC and q2 = -2 μC are placed at points A and B, separated by 6 cm. Find the magnitude of the electric force acting on q1 by q2? (Take k = 9×10^9 N·m²/C²) | Two point charges q1 = 3 μC and q2 = -2 μC are placed at points A and B, separated by 6 cm. Find the magnitude of the electric force acting on q1 by q2? (Take k = 9×10^9 N×m²/C²) | kept verified question correction |
| LD053 | LD | electrostatics_field | At two points A and B, separated by 10 cm in the air, two charges q1 = -q2 = 6 × 10^-6 C are placed. Determine the electric field strength caused by these two point charges at p... | At two points A and B, separated by 10 cm in the air, two charges q1 = -q2 = 6 × 10^-6 C are placed. Determine the electric field strength caused by these two point charges at p... | kept verified question correction |
| LD054 | LD | electrostatics_force | Two point charges, q1 = 6 × 10^-6 C and q2 = -6 × 10^-6 C, are placed in air at points A and B, separated by 10 cm.  Determine the electric field strength caused by these two po... | Two point charges, q1 = 6 × 10^-6 C and q2 = -6 × 10^-6 C, are placed in air at points A and B, separated by 10 cm.  Calculate the electric force exerted on a charge q3 = -3 × 1... | kept verified question correction |
| LD057 | LD | electrostatics_force | At two points A and B, 20 cm apart in the air, two charges q1 = 4 × 10^-6 C and q2 = -6.4 × 10^-6 C are placed. Determine the electric field strength caused by these two charges... | At two points A and B, 20 cm apart in the air, two charges q1 = 4 × 10^-6 C and q2 = -6.4 × 10^-6 C are placed. Determine the electric force acting on a charge q3 = -5 × 10^-8 C... | kept verified question correction |
| LD071 | LD | electrostatics_field | Two point charges q1 = -10^-6 C and q2 = 10^-6 C are placed at two points A and B, 40 cm apart in a vacuum. What is the magnitude of the resultant electric field at point N, whi... | Two point charges q1 = -10^-6 C and q2 = 10^-6 C are placed at points A and B, 40 cm apart in a vacuum. What is the magnitude of the resultant electric field at point N, which i... | kept verified question correction |
| NL393 | NL | capacitor | A capacitor has a capacitance of 6 µF. The voltage at time t is V(t) = 100 · cos(2000t) V. Calculate the energy stored in the electric field at t = 0.0015 s. | A capacitor has a capacitance of 6 µF. The voltage at time t is V(t) = 100 × cos(2000t) V. Calculate the energy stored in the electric field at t = 0.0015 s. | kept verified question correction |
| NL394 | NL | induction | A coil has an instantaneous current I(t) = 2.5 · cos(1500t) A and an inductance L = 0.3 H. Calculate the magnetic field energy stored in the coil at t = 0.001 s. | A coil has an instantaneous current I(t) = 2.5 × cos(1500t) A and an inductance L = 0.3 H. Calculate the magnetic field energy stored in the coil at t = 0.001 s. | kept verified question correction |
| NL399 | NL | capacitor | A capacitor has a voltage function U(t) = 250 · sin(1000t) V and a capacitance C = 8 µF. Calculate the maximum electric field energy stored in the capacitor. | A capacitor has a voltage function U(t) = 250 × sin(1000t) V and a capacitance C = 8 µF. Calculate the maximum electric field energy stored in the capacitor. | kept verified question correction |
| NL400 | NL | induction | A coil has an inductance L = 0.4 H and a current varying according to I(t) = 3 · sin(1000t) A. Calculate the maximum magnetic field energy stored in the coil. | A coil has an inductance L = 0.4 H and a current varying according to I(t) = 3 × sin(1000t) A. Calculate the maximum magnetic field energy stored in the coil. | kept verified question correction |
| TD016 | TD | capacitor | Two capacitors with capacitances C1 = 0.4μF and C2 = 0.6μF are connected in parallel, and then connected to a power source with a voltage U < 60 V. One of the two capacitors has... | Two capacitors with capacitances C1 = 0.4μF and C2 = 0.6μF are connected in parallel, and then connected to a power source with a voltage U < 60 V. One of the two capacitors has... | kept verified question correction |
| TD179 | TD | capacitor | A parallel-plate air capacitor has a capacitance of 36.53 pF and is charged at a potential difference of 124.5 V. Calculate the electric field energy stored in the capacitor. Gi... | A parallel-plate air capacitor has a capacitance of 36.53 pF and is charged at a potential difference of 124.5 V. Calculate the electric field energy stored in the capacitor. Gi... | kept verified question correction |
| TD357 | TD | capacitor | Here are a few ways to translate it, all being accurate and commonly used in physics:     1. **A capacitor with a capacitance of 5 μF is connected to a 10 V voltage. Calculate t... | A capacitor with a capacitance of 5 μF is connected to a 10 V voltage source. Calculate the charge on the capacitor. | kept verified question correction |

## Augmented Rows Kept
| id | prefix | topic | question | answer | unit | logged |
| --- | --- | --- | --- | --- | --- | --- |
| CH381 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 80 V and has resistance R = 20 Ω. Calculate the power consumed by the circuit. | 320 | W | yes |
| CH382 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 100 V and has resistance R = 30 Ω. Calculate the power consumed by the circuit. | 333.333 | W | yes |
| CH383 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 120 V and has resistance R = 40 Ω. Calculate the power consumed by the circuit. | 360 | W | yes |
| CH384 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 140 V and has resistance R = 50 Ω. Calculate the power consumed by the circuit. | 392 | W | yes |
| CH385 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 160 V and has resistance R = 60 Ω. Calculate the power consumed by the circuit. | 426.667 | W | yes |
| CH386 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 180 V and has resistance R = 20 Ω. Calculate the power consumed by the circuit. | 1620 | W | yes |
| CH387 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 200 V and has resistance R = 30 Ω. Calculate the power consumed by the circuit. | 1333.333 | W | yes |
| CH388 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 80 V and has resistance R = 40 Ω. Calculate the power consumed by the circuit. | 160 | W | yes |
| CH389 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 100 V and has resistance R = 50 Ω. Calculate the power consumed by the circuit. | 200 | W | yes |
| CH390 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 120 V and has resistance R = 60 Ω. Calculate the power consumed by the circuit. | 240 | W | yes |
| CH391 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 140 V and has resistance R = 20 Ω. Calculate the power consumed by the circuit. | 980 | W | yes |
| CH392 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 160 V and has resistance R = 30 Ω. Calculate the power consumed by the circuit. | 853.333 | W | yes |
| CH393 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 180 V and has resistance R = 40 Ω. Calculate the power consumed by the circuit. | 810 | W | yes |
| CH394 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 200 V and has resistance R = 50 Ω. Calculate the power consumed by the circuit. | 800 | W | yes |
| CH395 | CH | circuit_power | At resonance, an RLC series circuit is supplied with RMS voltage U = 80 V and has resistance R = 60 Ω. Calculate the power consumed by the circuit. | 106.667 | W | yes |
| CH396 | CH | ac_resonance | An RLC circuit has L = 0.05 H and C = 50 μF. Calculate the resonant frequency. | 100.66 | Hz | yes |
| CH397 | CH | ac_resonance | An RLC circuit has L = 0.1 H and C = 60 μF. Calculate the resonant frequency. | 64.97 | Hz | yes |
| CH398 | CH | ac_resonance | An RLC circuit has L = 0.15 H and C = 70 μF. Calculate the resonant frequency. | 49.12 | Hz | yes |
| CH399 | CH | ac_resonance | An RLC circuit has L = 0.2 H and C = 20 μF. Calculate the resonant frequency. | 79.58 | Hz | yes |
| CH400 | CH | ac_resonance | An RLC circuit has L = 0.25 H and C = 30 μF. Calculate the resonant frequency. | 58.12 | Hz | yes |
| CH401 | CH | ac_resonance | An RLC circuit has L = 0.05 H and C = 40 μF. Calculate the resonant frequency. | 112.54 | Hz | yes |
| CHLT021 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.5 H, and C = 20 μF. If the AC source frequency is 50.33 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT022 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.2 H, and C = 50 μF. If the AC source frequency is 50.33 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT023 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.1 H, and C = 25 μF. If the AC source frequency is 100.66 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT024 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.4 H, and C = 10 μF. If the AC source frequency is 79.58 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT025 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.3 H, and C = 30 μF. If the AC source frequency is 53.05 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT026 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.08 H, and C = 80 μF. If the AC source frequency is 62.91 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT027 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.5 H, and C = 20 μF. If the AC source frequency is 50.33 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT028 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.2 H, and C = 50 μF. If the AC source frequency is 50.33 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT029 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.1 H, and C = 25 μF. If the AC source frequency is 100.66 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT030 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.4 H, and C = 10 μF. If the AC source frequency is 79.58 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT031 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.3 H, and C = 30 μF. If the AC source frequency is 53.05 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT032 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.08 H, and C = 80 μF. If the AC source frequency is 62.91 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT033 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.5 H, and C = 20 μF. If the AC source frequency is 50.33 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT034 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.2 H, and C = 50 μF. If the AC source frequency is 50.33 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT035 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.1 H, and C = 25 μF. If the AC source frequency is 100.66 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT036 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.4 H, and C = 10 μF. If the AC source frequency is 79.58 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT037 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.3 H, and C = 30 μF. If the AC source frequency is 53.05 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT038 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.08 H, and C = 80 μF. If the AC source frequency is 62.91 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT039 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.5 H, and C = 20 μF. If the AC source frequency is 50.33 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT040 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.2 H, and C = 50 μF. If the AC source frequency is 50.33 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT041 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.1 H, and C = 25 μF. If the AC source frequency is 100.66 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT042 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.4 H, and C = 10 μF. If the AC source frequency is 79.58 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT043 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.3 H, and C = 30 μF. If the AC source frequency is 53.05 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT044 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.08 H, and C = 80 μF. If the AC source frequency is 62.91 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT045 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.5 H, and C = 20 μF. If the AC source frequency is 50.33 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT046 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.2 H, and C = 50 μF. If the AC source frequency is 50.33 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT047 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.1 H, and C = 25 μF. If the AC source frequency is 100.66 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT048 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.4 H, and C = 10 μF. If the AC source frequency is 79.58 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT049 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.3 H, and C = 30 μF. If the AC source frequency is 53.05 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT050 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.08 H, and C = 80 μF. If the AC source frequency is 62.91 Hz, does the circuit reach electrical resonance? | Yes | - | yes |
| CHLT051 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.5 H, and C = 20 μF. If the AC source frequency is 59.39 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT052 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.2 H, and C = 50 μF. If the AC source frequency is 59.39 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT053 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.1 H, and C = 25 μF. If the AC source frequency is 118.78 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT054 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.4 H, and C = 10 μF. If the AC source frequency is 93.9 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT055 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.3 H, and C = 30 μF. If the AC source frequency is 62.6 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT056 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.08 H, and C = 80 μF. If the AC source frequency is 74.24 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT057 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.5 H, and C = 20 μF. If the AC source frequency is 59.39 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT058 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.2 H, and C = 50 μF. If the AC source frequency is 59.39 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT059 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.1 H, and C = 25 μF. If the AC source frequency is 118.78 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT060 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.4 H, and C = 10 μF. If the AC source frequency is 93.9 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT061 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.3 H, and C = 30 μF. If the AC source frequency is 62.6 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT062 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.08 H, and C = 80 μF. If the AC source frequency is 74.24 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT063 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.5 H, and C = 20 μF. If the AC source frequency is 59.39 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT064 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.2 H, and C = 50 μF. If the AC source frequency is 59.39 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT065 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.1 H, and C = 25 μF. If the AC source frequency is 118.78 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT066 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.4 H, and C = 10 μF. If the AC source frequency is 93.9 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT067 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.3 H, and C = 30 μF. If the AC source frequency is 62.6 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT068 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.08 H, and C = 80 μF. If the AC source frequency is 74.24 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT069 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.5 H, and C = 20 μF. If the AC source frequency is 59.39 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT070 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.2 H, and C = 50 μF. If the AC source frequency is 59.39 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT071 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.1 H, and C = 25 μF. If the AC source frequency is 118.78 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT072 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.4 H, and C = 10 μF. If the AC source frequency is 93.9 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT073 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.3 H, and C = 30 μF. If the AC source frequency is 62.6 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT074 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.08 H, and C = 80 μF. If the AC source frequency is 74.24 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT075 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.5 H, and C = 20 μF. If the AC source frequency is 59.39 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT076 | CHLT | ac_resonance | An RLC series circuit has R = 20 Ω, L = 0.2 H, and C = 50 μF. If the AC source frequency is 59.39 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT077 | CHLT | ac_resonance | An RLC series circuit has R = 30 Ω, L = 0.1 H, and C = 25 μF. If the AC source frequency is 118.78 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT078 | CHLT | ac_resonance | An RLC series circuit has R = 40 Ω, L = 0.4 H, and C = 10 μF. If the AC source frequency is 93.9 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT079 | CHLT | ac_resonance | An RLC series circuit has R = 50 Ω, L = 0.3 H, and C = 30 μF. If the AC source frequency is 62.6 Hz, does the circuit reach electrical resonance? | No | - | yes |
| CHLT080 | CHLT | ac_resonance | An RLC series circuit has R = 60 Ω, L = 0.08 H, and C = 80 μF. If the AC source frequency is 74.24 Hz, does the circuit reach electrical resonance? | No | - | yes |
| DDT401 | DDT | induction | A long solenoid has a turn density n = 800 turns/m and carries a current I = 0.8 A. Calculate the magnetic field inside the solenoid. | 0.000804 | T | yes |
| DDT402 | DDT | induction | A long solenoid has a turn density n = 1000 turns/m and carries a current I = 1.2 A. Calculate the magnetic field inside the solenoid. | 0.001508 | T | yes |
| DDT403 | DDT | induction | A long solenoid has a turn density n = 1200 turns/m and carries a current I = 1.6 A. Calculate the magnetic field inside the solenoid. | 0.002413 | T | yes |
| DDT404 | DDT | induction | A long solenoid has a turn density n = 1400 turns/m and carries a current I = 2 A. Calculate the magnetic field inside the solenoid. | 0.003519 | T | yes |
| DDT405 | DDT | induction | A long solenoid has a turn density n = 1600 turns/m and carries a current I = 2.4 A. Calculate the magnetic field inside the solenoid. | 0.004825 | T | yes |
| DDT406 | DDT | induction | A long solenoid has a turn density n = 1800 turns/m and carries a current I = 2.8 A. Calculate the magnetic field inside the solenoid. | 0.006333 | T | yes |
| DDT407 | DDT | induction | A long solenoid has a turn density n = 2000 turns/m and carries a current I = 0.8 A. Calculate the magnetic field inside the solenoid. | 0.002011 | T | yes |
| DDT408 | DDT | induction | A long solenoid has a turn density n = 2200 turns/m and carries a current I = 1.2 A. Calculate the magnetic field inside the solenoid. | 0.003318 | T | yes |
| DDT409 | DDT | induction | A long solenoid has a turn density n = 800 turns/m and carries a current I = 1.6 A. Calculate the magnetic field inside the solenoid. | 0.001608 | T | yes |
| DDT410 | DDT | induction | A long solenoid has a turn density n = 1000 turns/m and carries a current I = 2 A. Calculate the magnetic field inside the solenoid. | 0.002513 | T | yes |
| DDT411 | DDT | induction | A long solenoid has a turn density n = 1200 turns/m and carries a current I = 2.4 A. Calculate the magnetic field inside the solenoid. | 0.003619 | T | yes |
| DDT412 | DDT | induction | A long solenoid has a turn density n = 1400 turns/m and carries a current I = 2.8 A. Calculate the magnetic field inside the solenoid. | 0.004926 | T | yes |
| DDT413 | DDT | induction | A long solenoid has a turn density n = 1600 turns/m and carries a current I = 0.8 A. Calculate the magnetic field inside the solenoid. | 0.001608 | T | yes |
| DDT414 | DDT | induction | A long solenoid has a turn density n = 1800 turns/m and carries a current I = 1.2 A. Calculate the magnetic field inside the solenoid. | 0.002714 | T | yes |
| DDT415 | DDT | induction | A long solenoid has a turn density n = 2000 turns/m and carries a current I = 1.6 A. Calculate the magnetic field inside the solenoid. | 0.004021 | T | yes |
| DDT416 | DDT | induction | A long solenoid has a turn density n = 2200 turns/m and carries a current I = 2 A. Calculate the magnetic field inside the solenoid. | 0.005529 | T | yes |
| DDT417 | DDT | induction | A long solenoid has a turn density n = 800 turns/m and carries a current I = 2.4 A. Calculate the magnetic field inside the solenoid. | 0.002413 | T | yes |
| DDT418 | DDT | induction | A long solenoid has a turn density n = 1000 turns/m and carries a current I = 2.8 A. Calculate the magnetic field inside the solenoid. | 0.003519 | T | yes |
| DDT419 | DDT | induction | A long solenoid has a turn density n = 1200 turns/m and carries a current I = 0.8 A. Calculate the magnetic field inside the solenoid. | 0.001206 | T | yes |
| DDT420 | DDT | induction | A long solenoid has a turn density n = 1400 turns/m and carries a current I = 1.2 A. Calculate the magnetic field inside the solenoid. | 0.002111 | T | yes |
| DDT421 | DDT | induction | A long solenoid has a turn density n = 1600 turns/m and carries a current I = 1.6 A. Calculate the magnetic field inside the solenoid. | 0.003217 | T | yes |
| DDT422 | DDT | induction | A long solenoid has a turn density n = 1800 turns/m and carries a current I = 2 A. Calculate the magnetic field inside the solenoid. | 0.004524 | T | yes |
| DDT423 | DDT | induction | A long solenoid has a turn density n = 2000 turns/m and carries a current I = 2.4 A. Calculate the magnetic field inside the solenoid. | 0.006032 | T | yes |
| DDT424 | DDT | induction | A long solenoid has a turn density n = 2200 turns/m and carries a current I = 2.8 A. Calculate the magnetic field inside the solenoid. | 0.007741 | T | yes |
| DDT425 | DDT | induction | A long solenoid has a turn density n = 800 turns/m and carries a current I = 0.8 A. Calculate the magnetic field inside the solenoid. | 0.000804 | T | yes |
| DDT426 | DDT | induction | A long solenoid has a turn density n = 1000 turns/m and carries a current I = 1.2 A. Calculate the magnetic field inside the solenoid. | 0.001508 | T | yes |
| DDT427 | DDT | induction | A long solenoid has a turn density n = 1200 turns/m and carries a current I = 1.6 A. Calculate the magnetic field inside the solenoid. | 0.002413 | T | yes |
| DDT428 | DDT | induction | A long solenoid has a turn density n = 1400 turns/m and carries a current I = 2 A. Calculate the magnetic field inside the solenoid. | 0.003519 | T | yes |
| DDT429 | DDT | induction | A long solenoid has a turn density n = 1600 turns/m and carries a current I = 2.4 A. Calculate the magnetic field inside the solenoid. | 0.004825 | T | yes |
| DDT430 | DDT | induction | A long solenoid has a turn density n = 1800 turns/m and carries a current I = 2.8 A. Calculate the magnetic field inside the solenoid. | 0.006333 | T | yes |
| DDT431 | DDT | LC_oscillation | An LC circuit has a capacitor C = 10 μF initially charged to U = 20 V and an inductor L = 0.05 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.002 | J | yes |
| DDT432 | DDT | LC_oscillation | An LC circuit has a capacitor C = 15 μF initially charged to U = 30 V and an inductor L = 0.08 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.00675 | J | yes |
| DDT433 | DDT | LC_oscillation | An LC circuit has a capacitor C = 20 μF initially charged to U = 40 V and an inductor L = 0.11 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.016 | J | yes |
| DDT434 | DDT | LC_oscillation | An LC circuit has a capacitor C = 25 μF initially charged to U = 50 V and an inductor L = 0.14 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.03125 | J | yes |
| DDT435 | DDT | LC_oscillation | An LC circuit has a capacitor C = 30 μF initially charged to U = 60 V and an inductor L = 0.17 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.054 | J | yes |
| DDT436 | DDT | LC_oscillation | An LC circuit has a capacitor C = 35 μF initially charged to U = 20 V and an inductor L = 0.05 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.007 | J | yes |
| DDT437 | DDT | LC_oscillation | An LC circuit has a capacitor C = 10 μF initially charged to U = 30 V and an inductor L = 0.08 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.0045 | J | yes |
| DDT438 | DDT | LC_oscillation | An LC circuit has a capacitor C = 15 μF initially charged to U = 40 V and an inductor L = 0.11 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.012 | J | yes |
| DDT439 | DDT | LC_oscillation | An LC circuit has a capacitor C = 20 μF initially charged to U = 50 V and an inductor L = 0.14 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.025 | J | yes |
| DDT440 | DDT | LC_oscillation | An LC circuit has a capacitor C = 25 μF initially charged to U = 60 V and an inductor L = 0.17 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.045 | J | yes |
| DDT441 | DDT | LC_oscillation | An LC circuit has a capacitor C = 30 μF initially charged to U = 20 V and an inductor L = 0.05 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.006 | J | yes |
| DDT442 | DDT | LC_oscillation | An LC circuit has a capacitor C = 35 μF initially charged to U = 30 V and an inductor L = 0.08 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.01575 | J | yes |
| DDT443 | DDT | LC_oscillation | An LC circuit has a capacitor C = 10 μF initially charged to U = 40 V and an inductor L = 0.11 H. Calculate the total electromagnetic energy of the ideal circuit. | 0.008 | J | yes |
| DT101 | DT | electrostatics_field | A point charge q = 2 nC is placed in air. Calculate the electric field strength at a point 3 cm from the charge. | 20000 | V/m | yes |
| DT102 | DT | electrostatics_field | A point charge q = 3 nC is placed in air. Calculate the electric field strength at a point 4 cm from the charge. | 16875 | V/m | yes |
| DT103 | DT | electrostatics_field | A point charge q = 4 nC is placed in air. Calculate the electric field strength at a point 5 cm from the charge. | 14400 | V/m | yes |
| DT104 | DT | electrostatics_field | A point charge q = 5 nC is placed in air. Calculate the electric field strength at a point 6 cm from the charge. | 12500 | V/m | yes |
| DT105 | DT | electrostatics_field | A point charge q = 6 nC is placed in air. Calculate the electric field strength at a point 7 cm from the charge. | 11020.41 | V/m | yes |
| DT106 | DT | electrostatics_field | A point charge q = 7 nC is placed in air. Calculate the electric field strength at a point 8 cm from the charge. | 9843.75 | V/m | yes |
| DT107 | DT | electrostatics_field | A point charge q = 8 nC is placed in air. Calculate the electric field strength at a point 9 cm from the charge. | 8888.89 | V/m | yes |
| DT108 | DT | electrostatics_field | A point charge q = 9 nC is placed in air. Calculate the electric field strength at a point 3 cm from the charge. | 90000 | V/m | yes |
| DT109 | DT | electrostatics_field | A point charge q = 2 nC is placed in air. Calculate the electric field strength at a point 4 cm from the charge. | 11250 | V/m | yes |
| DT110 | DT | electrostatics_field | A point charge q = 3 nC is placed in air. Calculate the electric field strength at a point 5 cm from the charge. | 10800 | V/m | yes |
| DT111 | DT | electrostatics_field | A point charge q = 4 nC is placed in air. Calculate the electric field strength at a point 6 cm from the charge. | 10000 | V/m | yes |
| DT112 | DT | electrostatics_field | A point charge q = 5 nC is placed in air. Calculate the electric field strength at a point 7 cm from the charge. | 9183.67 | V/m | yes |
| DT113 | DT | electrostatics_field | A point charge q = 6 nC is placed in air. Calculate the electric field strength at a point 8 cm from the charge. | 8437.5 | V/m | yes |
| DT114 | DT | electrostatics_field | A point charge q = 7 nC is placed in air. Calculate the electric field strength at a point 9 cm from the charge. | 7777.78 | V/m | yes |
| DT115 | DT | electrostatics_field | A point charge q = 8 nC is placed in air. Calculate the electric field strength at a point 3 cm from the charge. | 80000 | V/m | yes |
| DT116 | DT | electrostatics_field | A point charge q = 9 nC is placed in air. Calculate the electric field strength at a point 4 cm from the charge. | 50625 | V/m | yes |
| DT117 | DT | electrostatics_field | A point charge q = 2 nC is placed in air. Calculate the electric field strength at a point 5 cm from the charge. | 7200 | V/m | yes |
| DT118 | DT | electrostatics_field | A point charge q = 3 nC is placed in air. Calculate the electric field strength at a point 6 cm from the charge. | 7500 | V/m | yes |
| DT119 | DT | electrostatics_field | A point charge q = 4 nC is placed in air. Calculate the electric field strength at a point 7 cm from the charge. | 7346.94 | V/m | yes |
| DT120 | DT | electrostatics_field | A point charge q = 5 nC is placed in air. Calculate the electric field strength at a point 8 cm from the charge. | 7031.25 | V/m | yes |
| DT121 | DT | electrostatics_field | A point charge q = 6 nC is placed in air. Calculate the electric field strength at a point 9 cm from the charge. | 6666.67 | V/m | yes |
| DT122 | DT | electrostatics_field | A point charge q = 7 nC is placed in air. Calculate the electric field strength at a point 3 cm from the charge. | 70000 | V/m | yes |
| DT123 | DT | electrostatics_field | A point charge q = 8 nC is placed in air. Calculate the electric field strength at a point 4 cm from the charge. | 45000 | V/m | yes |
| DT124 | DT | electrostatics_field | A point charge q = 9 nC is placed in air. Calculate the electric field strength at a point 5 cm from the charge. | 32400 | V/m | yes |
| DT125 | DT | electrostatics_field | A point charge q = 2 nC is placed in air. Calculate the electric field strength at a point 6 cm from the charge. | 5000 | V/m | yes |
| DT126 | DT | electrostatics_field | A point charge q = 3 nC is placed in air. Calculate the electric field strength at a point 7 cm from the charge. | 5510.2 | V/m | yes |
| DT127 | DT | electrostatics_field | A point charge q = 4 nC is placed in air. Calculate the electric field strength at a point 8 cm from the charge. | 5625 | V/m | yes |
| DT128 | DT | electrostatics_field | A point charge q = 5 nC is placed in air. Calculate the electric field strength at a point 9 cm from the charge. | 5555.56 | V/m | yes |
| DT129 | DT | electrostatics_field | A point charge q = 6 nC is placed in air. Calculate the electric field strength at a point 3 cm from the charge. | 60000 | V/m | yes |
| DT130 | DT | electrostatics_field | A point charge q = 7 nC is placed in air. Calculate the electric field strength at a point 4 cm from the charge. | 39375 | V/m | yes |
| DT131 | DT | electrostatics_field | A point charge q = 8 nC is placed in air. Calculate the electric field strength at a point 5 cm from the charge. | 28800 | V/m | yes |
| DT132 | DT | electrostatics_field | A point charge q = 9 nC is placed in air. Calculate the electric field strength at a point 6 cm from the charge. | 22500 | V/m | yes |
| DT133 | DT | electrostatics_field | A point charge q = 2 nC is placed in air. Calculate the electric field strength at a point 7 cm from the charge. | 3673.47 | V/m | yes |
| DT134 | DT | electrostatics_field | A point charge q = 3 nC is placed in air. Calculate the electric field strength at a point 8 cm from the charge. | 4218.75 | V/m | yes |
| DT135 | DT | electrostatics_field | A point charge q = 4 nC is placed in air. Calculate the electric field strength at a point 9 cm from the charge. | 4444.44 | V/m | yes |
| DT136 | DT | electrostatics_field | A point charge q = 5 nC is placed in air. Calculate the electric field strength at a point 3 cm from the charge. | 50000 | V/m | yes |
| DT137 | DT | electrostatics_field | A point charge q = 6 nC is placed in air. Calculate the electric field strength at a point 4 cm from the charge. | 33750 | V/m | yes |
| DT138 | DT | electrostatics_field | A point charge q = 7 nC is placed in air. Calculate the electric field strength at a point 5 cm from the charge. | 25200 | V/m | yes |
| DT139 | DT | electrostatics_field | A point charge q = 8 nC is placed in air. Calculate the electric field strength at a point 6 cm from the charge. | 20000 | V/m | yes |
| DT140 | DT | electrostatics_field | A point charge q = 9 nC is placed in air. Calculate the electric field strength at a point 7 cm from the charge. | 16530.61 | V/m | yes |
| DT141 | DT | electrostatics_field | A point charge q = 2 nC is placed in air. Calculate the electric field strength at a point 8 cm from the charge. | 2812.5 | V/m | yes |
| DT142 | DT | electrostatics_field | A point charge q = 3 nC is placed in air. Calculate the electric field strength at a point 9 cm from the charge. | 3333.33 | V/m | yes |
| DT143 | DT | electrostatics_field | A point charge q = 4 nC is placed in air. Calculate the electric field strength at a point 3 cm from the charge. | 40000 | V/m | yes |
| DT144 | DT | electrostatics_field | A point charge q = 5 nC is placed in air. Calculate the electric field strength at a point 4 cm from the charge. | 28125 | V/m | yes |
| DT145 | DT | electrostatics_field | A point charge q = 6 nC is placed in air. Calculate the electric field strength at a point 5 cm from the charge. | 21600 | V/m | yes |
| DT146 | DT | electrostatics_field | A point charge q = 7 nC is placed in air. Calculate the electric field strength at a point 6 cm from the charge. | 17500 | V/m | yes |
| DT147 | DT | electrostatics_field | A point charge q = 8 nC is placed in air. Calculate the electric field strength at a point 7 cm from the charge. | 14693.88 | V/m | yes |
| DT148 | DT | electrostatics_field | A point charge q = 9 nC is placed in air. Calculate the electric field strength at a point 8 cm from the charge. | 12656.25 | V/m | yes |
| DT149 | DT | electrostatics_field | A point charge q = 2 nC is placed in air. Calculate the electric field strength at a point 9 cm from the charge. | 2222.22 | V/m | yes |
| DT150 | DT | electrostatics_field | A point charge q = 3 nC is placed in air. Calculate the electric field strength at a point 3 cm from the charge. | 30000 | V/m | yes |
| DT151 | DT | electrostatics_force | A charge q = 3 nC is placed in a uniform electric field of magnitude 19500 V/m. Calculate the magnitude of the electric force acting on the charge. | 58.5 x 10^-6 | N | yes |
| DT152 | DT | electrostatics_force | A charge q = 4 nC is placed in a uniform electric field of magnitude 21000 V/m. Calculate the magnitude of the electric force acting on the charge. | 84 x 10^-6 | N | yes |
| DT153 | DT | electrostatics_force | A charge q = 5 nC is placed in a uniform electric field of magnitude 22500 V/m. Calculate the magnitude of the electric force acting on the charge. | 112.5 x 10^-6 | N | yes |
| DT154 | DT | electrostatics_force | A charge q = 6 nC is placed in a uniform electric field of magnitude 24000 V/m. Calculate the magnitude of the electric force acting on the charge. | 144 x 10^-6 | N | yes |
| DT155 | DT | electrostatics_force | A charge q = 7 nC is placed in a uniform electric field of magnitude 12000 V/m. Calculate the magnitude of the electric force acting on the charge. | 84 x 10^-6 | N | yes |
| DT156 | DT | electrostatics_force | A charge q = 3 nC is placed in a uniform electric field of magnitude 13500 V/m. Calculate the magnitude of the electric force acting on the charge. | 40.5 x 10^-6 | N | yes |
| DT157 | DT | electrostatics_force | A charge q = 4 nC is placed in a uniform electric field of magnitude 15000 V/m. Calculate the magnitude of the electric force acting on the charge. | 60 x 10^-6 | N | yes |
| DT158 | DT | electrostatics_force | A charge q = 5 nC is placed in a uniform electric field of magnitude 16500 V/m. Calculate the magnitude of the electric force acting on the charge. | 82.5 x 10^-6 | N | yes |
| DT159 | DT | electrostatics_force | A charge q = 6 nC is placed in a uniform electric field of magnitude 18000 V/m. Calculate the magnitude of the electric force acting on the charge. | 108 x 10^-6 | N | yes |
| DT160 | DT | electrostatics_force | A charge q = 7 nC is placed in a uniform electric field of magnitude 19500 V/m. Calculate the magnitude of the electric force acting on the charge. | 136.5 x 10^-6 | N | yes |
| DT161 | DT | electrostatics_force | A charge q = 3 nC is placed in a uniform electric field of magnitude 21000 V/m. Calculate the magnitude of the electric force acting on the charge. | 63 x 10^-6 | N | yes |
| DT162 | DT | electrostatics_force | A charge q = 4 nC is placed in a uniform electric field of magnitude 22500 V/m. Calculate the magnitude of the electric force acting on the charge. | 90 x 10^-6 | N | yes |
| DT163 | DT | electrostatics_force | A charge q = 5 nC is placed in a uniform electric field of magnitude 24000 V/m. Calculate the magnitude of the electric force acting on the charge. | 120 x 10^-6 | N | yes |
| LD401 | LD | electrostatics_force | Two point charges q1 = 1 μC and q2 = 2 μC are separated by 5 cm in air. Calculate the magnitude of the electrostatic force between them. | 7.2 | N | yes |
| LD402 | LD | electrostatics_force | Two point charges q1 = 2 μC and q2 = 3 μC are separated by 6 cm in air. Calculate the magnitude of the electrostatic force between them. | 15 | N | yes |
| LD403 | LD | electrostatics_force | Two point charges q1 = 3 μC and q2 = 4 μC are separated by 7 cm in air. Calculate the magnitude of the electrostatic force between them. | 22.041 | N | yes |
| LD404 | LD | electrostatics_force | Two point charges q1 = 4 μC and q2 = 5 μC are separated by 8 cm in air. Calculate the magnitude of the electrostatic force between them. | 28.125 | N | yes |
| LD405 | LD | electrostatics_force | Two point charges q1 = 5 μC and q2 = 6 μC are separated by 9 cm in air. Calculate the magnitude of the electrostatic force between them. | 33.333 | N | yes |
| LD406 | LD | electrostatics_force | Two point charges q1 = 1 μC and q2 = 7 μC are separated by 10 cm in air. Calculate the magnitude of the electrostatic force between them. | 6.3 | N | yes |
| LD407 | LD | electrostatics_force | Two point charges q1 = 2 μC and q2 = 2 μC are separated by 5 cm in air. Calculate the magnitude of the electrostatic force between them. | 14.4 | N | yes |
| LD408 | LD | electrostatics_force | Two point charges q1 = 3 μC and q2 = 3 μC are separated by 6 cm in air. Calculate the magnitude of the electrostatic force between them. | 22.5 | N | yes |
| LD409 | LD | electrostatics_force | Two point charges q1 = 4 μC and q2 = 4 μC are separated by 7 cm in air. Calculate the magnitude of the electrostatic force between them. | 29.388 | N | yes |
| LD410 | LD | electrostatics_force | Two point charges q1 = 5 μC and q2 = 5 μC are separated by 8 cm in air. Calculate the magnitude of the electrostatic force between them. | 35.156 | N | yes |
| LD411 | LD | electrostatics_force | Two point charges q1 = 1 μC and q2 = 6 μC are separated by 9 cm in air. Calculate the magnitude of the electrostatic force between them. | 6.667 | N | yes |
| LD412 | LD | electrostatics_force | Two point charges q1 = 2 μC and q2 = 7 μC are separated by 10 cm in air. Calculate the magnitude of the electrostatic force between them. | 12.6 | N | yes |
| LD413 | LD | electrostatics_force | Two point charges q1 = 3 μC and q2 = 2 μC are separated by 5 cm in air. Calculate the magnitude of the electrostatic force between them. | 21.6 | N | yes |
| LD414 | LD | electrostatics_force | Two point charges q1 = 4 μC and q2 = 3 μC are separated by 6 cm in air. Calculate the magnitude of the electrostatic force between them. | 30 | N | yes |
| LD415 | LD | electrostatics_force | Two point charges q1 = 5 μC and q2 = 4 μC are separated by 7 cm in air. Calculate the magnitude of the electrostatic force between them. | 36.735 | N | yes |
| LD416 | LD | electrostatics_force | Two point charges q1 = 1 μC and q2 = 5 μC are separated by 8 cm in air. Calculate the magnitude of the electrostatic force between them. | 7.031 | N | yes |
| NL401 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 5 μF is initially charged to 40 V. Calculate the total oscillation energy in mJ. | 4 | mJ | yes |
| NL402 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 10 μF is initially charged to 60 V. Calculate the total oscillation energy in mJ. | 18 | mJ | yes |
| NL403 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 15 μF is initially charged to 80 V. Calculate the total oscillation energy in mJ. | 48 | mJ | yes |
| NL404 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 20 μF is initially charged to 100 V. Calculate the total oscillation energy in mJ. | 100 | mJ | yes |
| NL405 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 25 μF is initially charged to 120 V. Calculate the total oscillation energy in mJ. | 180 | mJ | yes |
| NL406 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 30 μF is initially charged to 40 V. Calculate the total oscillation energy in mJ. | 24 | mJ | yes |
| NL407 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 35 μF is initially charged to 60 V. Calculate the total oscillation energy in mJ. | 63 | mJ | yes |
| NL408 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 40 μF is initially charged to 80 V. Calculate the total oscillation energy in mJ. | 128 | mJ | yes |
| NL409 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 5 μF is initially charged to 100 V. Calculate the total oscillation energy in mJ. | 25 | mJ | yes |
| NL410 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 10 μF is initially charged to 120 V. Calculate the total oscillation energy in mJ. | 72 | mJ | yes |
| NL411 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 15 μF is initially charged to 40 V. Calculate the total oscillation energy in mJ. | 12 | mJ | yes |
| NL412 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 20 μF is initially charged to 60 V. Calculate the total oscillation energy in mJ. | 36 | mJ | yes |
| NL413 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 25 μF is initially charged to 80 V. Calculate the total oscillation energy in mJ. | 80 | mJ | yes |
| NL414 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 30 μF is initially charged to 100 V. Calculate the total oscillation energy in mJ. | 150 | mJ | yes |
| NL415 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 35 μF is initially charged to 120 V. Calculate the total oscillation energy in mJ. | 252 | mJ | yes |
| NL416 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 40 μF is initially charged to 40 V. Calculate the total oscillation energy in mJ. | 32 | mJ | yes |
| NL417 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 5 μF is initially charged to 60 V. Calculate the total oscillation energy in mJ. | 9 | mJ | yes |
| NL418 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 10 μF is initially charged to 80 V. Calculate the total oscillation energy in mJ. | 32 | mJ | yes |
| NL419 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 15 μF is initially charged to 100 V. Calculate the total oscillation energy in mJ. | 75 | mJ | yes |
| NL420 | NL | LC_oscillation | In an ideal LC circuit, a capacitor C = 20 μF is initially charged to 120 V. Calculate the total oscillation energy in mJ. | 144 | mJ | yes |
| NL421 | NL | induction | An inductor with inductance L = 0.2 H carries a current I = 1 A. Calculate the magnetic field energy in mJ. | 100 | mJ | yes |
| NL422 | NL | induction | An inductor with inductance L = 0.25 H carries a current I = 1.5 A. Calculate the magnetic field energy in mJ. | 281.25 | mJ | yes |
| NL423 | NL | induction | An inductor with inductance L = 0.3 H carries a current I = 2 A. Calculate the magnetic field energy in mJ. | 600 | mJ | yes |
| NL424 | NL | induction | An inductor with inductance L = 0.35 H carries a current I = 2.5 A. Calculate the magnetic field energy in mJ. | 1093.75 | mJ | yes |
| NL425 | NL | induction | An inductor with inductance L = 0.1 H carries a current I = 3 A. Calculate the magnetic field energy in mJ. | 450 | mJ | yes |
| NL426 | NL | induction | An inductor with inductance L = 0.15 H carries a current I = 1 A. Calculate the magnetic field energy in mJ. | 75 | mJ | yes |
| NL427 | NL | capacitor | A capacitor with capacitance C = 20 μF is charged to U = 75 V. Calculate the electric field energy in mJ. | 56.25 | mJ | yes |
| NL428 | NL | capacitor | A capacitor with capacitance C = 30 μF is charged to U = 100 V. Calculate the electric field energy in mJ. | 150 | mJ | yes |
| NL429 | NL | capacitor | A capacitor with capacitance C = 40 μF is charged to U = 125 V. Calculate the electric field energy in mJ. | 312.5 | mJ | yes |
| NL430 | NL | capacitor | A capacitor with capacitance C = 50 μF is charged to U = 150 V. Calculate the electric field energy in mJ. | 562.5 | mJ | yes |
| NL431 | NL | capacitor | A capacitor with capacitance C = 10 μF is charged to U = 50 V. Calculate the electric field energy in mJ. | 12.5 | mJ | yes |
| TD403 | TD | capacitor | A capacitor stores a charge Q = 20 μC under a voltage U = 5 V. Calculate its capacitance in μF. | 4 | μF | yes |
| TD404 | TD | capacitor | A capacitor has capacitance C = 10 μF and is connected to a voltage U = 20 V. Calculate the charge on the capacitor in μC. | 200 | μC | yes |
| TD405 | TD | capacitor | A capacitor stores a charge Q = 40 μC under a voltage U = 15 V. Calculate its capacitance in μF. | 2.667 | μF | yes |
| TD406 | TD | capacitor | A capacitor has capacitance C = 20 μF and is connected to a voltage U = 40 V. Calculate the charge on the capacitor in μC. | 800 | μC | yes |
| TD407 | TD | capacitor | A capacitor stores a charge Q = 60 μC under a voltage U = 25 V. Calculate its capacitance in μF. | 2.4 | μF | yes |
| TD408 | TD | capacitor | A capacitor has capacitance C = 30 μF and is connected to a voltage U = 60 V. Calculate the charge on the capacitor in μC. | 1800 | μC | yes |
| TD409 | TD | capacitor | A capacitor stores a charge Q = 80 μC under a voltage U = 5 V. Calculate its capacitance in μF. | 16 | μF | yes |
| TD410 | TD | capacitor | A capacitor has capacitance C = 40 μF and is connected to a voltage U = 20 V. Calculate the charge on the capacitor in μC. | 800 | μC | yes |
| TD411 | TD | capacitor | A capacitor stores a charge Q = 100 μC under a voltage U = 15 V. Calculate its capacitance in μF. | 6.667 | μF | yes |
| TD412 | TD | capacitor | A capacitor has capacitance C = 5 μF and is connected to a voltage U = 40 V. Calculate the charge on the capacitor in μC. | 200 | μC | yes |
| TD413 | TD | capacitor | A capacitor stores a charge Q = 20 μC under a voltage U = 25 V. Calculate its capacitance in μF. | 0.8 | μF | yes |
| TD414 | TD | capacitor | A capacitor has capacitance C = 15 μF and is connected to a voltage U = 60 V. Calculate the charge on the capacitor in μC. | 900 | μC | yes |
| TD415 | TD | capacitor | A capacitor stores a charge Q = 40 μC under a voltage U = 5 V. Calculate its capacitance in μF. | 8 | μF | yes |
| TD416 | TD | capacitor | A capacitor has capacitance C = 25 μF and is connected to a voltage U = 20 V. Calculate the charge on the capacitor in μC. | 500 | μC | yes |
| TD417 | TD | capacitor | A capacitor stores a charge Q = 60 μC under a voltage U = 15 V. Calculate its capacitance in μF. | 4 | μF | yes |
| TD418 | TD | capacitor | A capacitor has capacitance C = 35 μF and is connected to a voltage U = 40 V. Calculate the charge on the capacitor in μC. | 1400 | μC | yes |
| TD419 | TD | capacitor | A capacitor stores a charge Q = 80 μC under a voltage U = 25 V. Calculate its capacitance in μF. | 3.2 | μF | yes |
| TD420 | TD | capacitor | A capacitor has capacitance C = 45 μF and is connected to a voltage U = 60 V. Calculate the charge on the capacitor in μC. | 2700 | μC | yes |
| TD421 | TD | capacitor | A capacitor stores a charge Q = 100 μC under a voltage U = 5 V. Calculate its capacitance in μF. | 20 | μF | yes |
| TD422 | TD | capacitor | A capacitor has capacitance C = 10 μF and is connected to a voltage U = 20 V. Calculate the charge on the capacitor in μC. | 200 | μC | yes |
| TD423 | TD | capacitor | A capacitor stores a charge Q = 20 μC under a voltage U = 15 V. Calculate its capacitance in μF. | 1.333 | μF | yes |
| TD424 | TD | capacitor | A capacitor has capacitance C = 20 μF and is connected to a voltage U = 40 V. Calculate the charge on the capacitor in μC. | 800 | μC | yes |
| TD425 | TD | capacitor | A capacitor stores a charge Q = 40 μC under a voltage U = 25 V. Calculate its capacitance in μF. | 1.6 | μF | yes |
| TD426 | TD | capacitor | A capacitor has capacitance C = 30 μF and is connected to a voltage U = 60 V. Calculate the charge on the capacitor in μC. | 1800 | μC | yes |
| TD427 | TD | capacitor | A capacitor stores a charge Q = 60 μC under a voltage U = 5 V. Calculate its capacitance in μF. | 12 | μF | yes |
| TD428 | TD | capacitor | A capacitor has capacitance C = 40 μF and is connected to a voltage U = 20 V. Calculate the charge on the capacitor in μC. | 800 | μC | yes |
| TD429 | TD | capacitor | A capacitor stores a charge Q = 80 μC under a voltage U = 15 V. Calculate its capacitance in μF. | 5.333 | μF | yes |
| TD430 | TD | capacitor | A capacitor has capacitance C = 5 μF and is connected to a voltage U = 40 V. Calculate the charge on the capacitor in μC. | 200 | μC | yes |
| TD431 | TD | capacitor | A capacitor stores a charge Q = 100 μC under a voltage U = 25 V. Calculate its capacitance in μF. | 4 | μF | yes |
| TD432 | TD | capacitor | A capacitor has capacitance C = 15 μF and is connected to a voltage U = 60 V. Calculate the charge on the capacitor in μC. | 900 | μC | yes |
| TD433 | TD | capacitor | A capacitor stores a charge Q = 20 μC under a voltage U = 5 V. Calculate its capacitance in μF. | 4 | μF | yes |
| THCB136 | THCB | measurement_error | A student measures a length as 50 cm with an absolute error of 0.1 cm. Calculate the percentage relative error. | 0.2 | % | yes |
| THCB137 | THCB | measurement_error | A student measures a length as 52 cm with an absolute error of 0.15 cm. Calculate the percentage relative error. | 0.288 | % | yes |
| THCB138 | THCB | measurement_error | A student measures a length as 54 cm with an absolute error of 0.2 cm. Calculate the percentage relative error. | 0.37 | % | yes |
| THCB139 | THCB | measurement_error | A student measures a length as 56 cm with an absolute error of 0.25 cm. Calculate the percentage relative error. | 0.446 | % | yes |
| THCB140 | THCB | measurement_error | A student measures a length as 58 cm with an absolute error of 0.3 cm. Calculate the percentage relative error. | 0.517 | % | yes |
| THCB141 | THCB | measurement_error | A student measures a length as 60 cm with an absolute error of 0.1 cm. Calculate the percentage relative error. | 0.167 | % | yes |
| THCB142 | THCB | measurement_error | A student measures a length as 62 cm with an absolute error of 0.15 cm. Calculate the percentage relative error. | 0.242 | % | yes |
| THCB143 | THCB | measurement_error | A student measures a length as 64 cm with an absolute error of 0.2 cm. Calculate the percentage relative error. | 0.312 | % | yes |
| THCB144 | THCB | measurement_error | A student measures a length as 66 cm with an absolute error of 0.25 cm. Calculate the percentage relative error. | 0.379 | % | yes |
| THCB145 | THCB | measurement_error | A student measures a length as 68 cm with an absolute error of 0.3 cm. Calculate the percentage relative error. | 0.441 | % | yes |
| THCB146 | THCB | measurement_error | A student measures a length as 70 cm with an absolute error of 0.1 cm. Calculate the percentage relative error. | 0.143 | % | yes |
| THCB147 | THCB | measurement_error | A student measures a length as 72 cm with an absolute error of 0.15 cm. Calculate the percentage relative error. | 0.208 | % | yes |
| THCB148 | THCB | measurement_error | A student measures a length as 74 cm with an absolute error of 0.2 cm. Calculate the percentage relative error. | 0.27 | % | yes |
| THCB149 | THCB | measurement_error | A student measures a length as 76 cm with an absolute error of 0.25 cm. Calculate the percentage relative error. | 0.329 | % | yes |
| THCB150 | THCB | measurement_error | A student measures a length as 78 cm with an absolute error of 0.3 cm. Calculate the percentage relative error. | 0.385 | % | yes |
| THCB151 | THCB | measurement_error | A student measures a length as 80 cm with an absolute error of 0.1 cm. Calculate the percentage relative error. | 0.125 | % | yes |
| THCB152 | THCB | measurement_error | A student measures a length as 82 cm with an absolute error of 0.15 cm. Calculate the percentage relative error. | 0.183 | % | yes |
| THCB153 | THCB | measurement_error | A student measures a length as 84 cm with an absolute error of 0.2 cm. Calculate the percentage relative error. | 0.238 | % | yes |
| THCB154 | THCB | measurement_error | A student measures a length as 86 cm with an absolute error of 0.25 cm. Calculate the percentage relative error. | 0.291 | % | yes |
| THCB155 | THCB | measurement_error | A student measures a length as 88 cm with an absolute error of 0.3 cm. Calculate the percentage relative error. | 0.341 | % | yes |
| THCB156 | THCB | measurement_error | A student measures a length as 90 cm with an absolute error of 0.1 cm. Calculate the percentage relative error. | 0.111 | % | yes |
| THCB157 | THCB | measurement_error | A student measures a length as 92 cm with an absolute error of 0.15 cm. Calculate the percentage relative error. | 0.163 | % | yes |
| THCB158 | THCB | measurement_error | A student measures a length as 94 cm with an absolute error of 0.2 cm. Calculate the percentage relative error. | 0.213 | % | yes |
| THCB159 | THCB | measurement_error | A student measures a length as 96 cm with an absolute error of 0.25 cm. Calculate the percentage relative error. | 0.26 | % | yes |
| THCB160 | THCB | measurement_error | A student measures a length as 98 cm with an absolute error of 0.3 cm. Calculate the percentage relative error. | 0.306 | % | yes |
| THCB161 | THCB | circuit_resistance | Two resistors R1 = 15 Ω and R2 = 20 Ω are connected in parallel to a 18 V source. Calculate the total current. | 2.1 | A | yes |
| THCB162 | THCB | circuit_resistance | Two resistors R1 = 20 Ω and R2 = 30 Ω are connected in parallel to a 24 V source. Calculate the total current. | 2 | A | yes |
| THCB163 | THCB | circuit_resistance | Two resistors R1 = 25 Ω and R2 = 40 Ω are connected in parallel to a 30 V source. Calculate the total current. | 1.95 | A | yes |
| THCB164 | THCB | circuit_resistance | Two resistors R1 = 30 Ω and R2 = 50 Ω are connected in parallel to a 12 V source. Calculate the total current. | 0.64 | A | yes |
| THCB165 | THCB | circuit_resistance | Two resistors R1 = 35 Ω and R2 = 60 Ω are connected in parallel to a 18 V source. Calculate the total current. | 0.814 | A | yes |
| THCB166 | THCB | circuit_resistance | Two resistors R1 = 10 Ω and R2 = 20 Ω are connected in parallel to a 24 V source. Calculate the total current. | 3.6 | A | yes |
| THCB167 | THCB | circuit_resistance | Two resistors R1 = 15 Ω and R2 = 30 Ω are connected in parallel to a 30 V source. Calculate the total current. | 3 | A | yes |
| THCB168 | THCB | circuit_resistance | Two resistors R1 = 20 Ω and R2 = 40 Ω are connected in parallel to a 12 V source. Calculate the total current. | 0.9 | A | yes |
| THCB169 | THCB | circuit_resistance | Two resistors R1 = 25 Ω and R2 = 50 Ω are connected in parallel to a 18 V source. Calculate the total current. | 1.08 | A | yes |
| THCB170 | THCB | circuit_resistance | Two resistors R1 = 30 Ω and R2 = 60 Ω are connected in parallel to a 24 V source. Calculate the total current. | 1.2 | A | yes |
| THCB171 | THCB | circuit_resistance | Two resistors R1 = 35 Ω and R2 = 20 Ω are connected in parallel to a 30 V source. Calculate the total current. | 2.357 | A | yes |
| THCB172 | THCB | circuit_resistance | Two resistors R1 = 10 Ω and R2 = 30 Ω are connected in parallel to a 12 V source. Calculate the total current. | 1.6 | A | yes |
| THCB173 | THCB | circuit_resistance | Two resistors R1 = 15 Ω and R2 = 40 Ω are connected in parallel to a 18 V source. Calculate the total current. | 1.65 | A | yes |
| THCB174 | THCB | circuit_resistance | Two resistors R1 = 20 Ω and R2 = 50 Ω are connected in parallel to a 24 V source. Calculate the total current. | 1.68 | A | yes |
| THCB175 | THCB | circuit_resistance | Two resistors R1 = 25 Ω and R2 = 60 Ω are connected in parallel to a 30 V source. Calculate the total current. | 1.7 | A | yes |
| THCB176 | THCB | circuit_resistance | Two resistors R1 = 30 Ω and R2 = 20 Ω are connected in parallel to a 12 V source. Calculate the total current. | 1 | A | yes |
| THCB177 | THCB | circuit_resistance | Two resistors R1 = 35 Ω and R2 = 30 Ω are connected in parallel to a 18 V source. Calculate the total current. | 1.114 | A | yes |
| THCB178 | THCB | circuit_resistance | Two resistors R1 = 10 Ω and R2 = 40 Ω are connected in parallel to a 24 V source. Calculate the total current. | 3 | A | yes |
| THCB179 | THCB | circuit_resistance | Two resistors R1 = 15 Ω and R2 = 50 Ω are connected in parallel to a 30 V source. Calculate the total current. | 2.6 | A | yes |
| THCB180 | THCB | circuit_resistance | Two resistors R1 = 20 Ω and R2 = 60 Ω are connected in parallel to a 12 V source. Calculate the total current. | 0.8 | A | yes |
| THCB181 | THCB | circuit_power | A resistor R = 9 Ω is connected across a voltage U = 15 V. Calculate the electric power dissipated by the resistor. | 25 | W | yes |
| THCB182 | THCB | circuit_power | A resistor R = 11 Ω is connected across a voltage U = 18 V. Calculate the electric power dissipated by the resistor. | 29.455 | W | yes |
| THCB183 | THCB | circuit_power | A resistor R = 13 Ω is connected across a voltage U = 21 V. Calculate the electric power dissipated by the resistor. | 33.923 | W | yes |
| THCB184 | THCB | circuit_power | A resistor R = 15 Ω is connected across a voltage U = 6 V. Calculate the electric power dissipated by the resistor. | 2.4 | W | yes |
| THCB185 | THCB | circuit_power | A resistor R = 3 Ω is connected across a voltage U = 9 V. Calculate the electric power dissipated by the resistor. | 27 | W | yes |
| THCB186 | THCB | circuit_power | A resistor R = 5 Ω is connected across a voltage U = 12 V. Calculate the electric power dissipated by the resistor. | 28.8 | W | yes |
| THCB187 | THCB | circuit_power | A resistor R = 7 Ω is connected across a voltage U = 15 V. Calculate the electric power dissipated by the resistor. | 32.143 | W | yes |
| THCB188 | THCB | circuit_power | A resistor R = 9 Ω is connected across a voltage U = 18 V. Calculate the electric power dissipated by the resistor. | 36 | W | yes |
| THCB189 | THCB | circuit_power | A resistor R = 11 Ω is connected across a voltage U = 21 V. Calculate the electric power dissipated by the resistor. | 40.091 | W | yes |
| THCB190 | THCB | circuit_power | A resistor R = 13 Ω is connected across a voltage U = 6 V. Calculate the electric power dissipated by the resistor. | 2.769 | W | yes |
| THCB191 | THCB | circuit_power | A resistor R = 15 Ω is connected across a voltage U = 9 V. Calculate the electric power dissipated by the resistor. | 5.4 | W | yes |
| CH402 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 90 V and resistance R = 30 Ω. Calculate the active power consumed by the circuit. | 270 | W | yes |
| CH403 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 110 V and resistance R = 55 Ω. Calculate the active power consumed by the circuit. | 220 | W | yes |
| CH404 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 150 V and resistance R = 75 Ω. Calculate the active power consumed by the circuit. | 300 | W | yes |
| CH405 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 180 V and resistance R = 45 Ω. Calculate the active power consumed by the circuit. | 720 | W | yes |
| CH406 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 220 V and resistance R = 80 Ω. Calculate the active power consumed by the circuit. | 605 | W | yes |
| CH407 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 75 V and resistance R = 25 Ω. Calculate the active power consumed by the circuit. | 225 | W | yes |
| CH408 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 125 V and resistance R = 50 Ω. Calculate the active power consumed by the circuit. | 312.5 | W | yes |
| CH409 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 160 V and resistance R = 64 Ω. Calculate the active power consumed by the circuit. | 400 | W | yes |
| CH410 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 200 V and resistance R = 100 Ω. Calculate the active power consumed by the circuit. | 400 | W | yes |
| CH411 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 240 V and resistance R = 120 Ω. Calculate the active power consumed by the circuit. | 480 | W | yes |
| CH412 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 140 V and resistance R = 35 Ω. Calculate the active power consumed by the circuit. | 560 | W | yes |
| CH413 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 210 V and resistance R = 70 Ω. Calculate the active power consumed by the circuit. | 630 | W | yes |
| CH414 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 96 V and resistance R = 24 Ω. Calculate the active power consumed by the circuit. | 384 | W | yes |
| CH415 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 144 V and resistance R = 36 Ω. Calculate the active power consumed by the circuit. | 576 | W | yes |
| CH416 | CH | circuit_power | At resonance, a series RLC circuit has RMS voltage U = 132 V and resistance R = 44 Ω. Calculate the active power consumed by the circuit. | 396 | W | yes |
| CH417 | CH | circuit_power | A series AC circuit has RMS voltage U = 100 V, resistance R = 36 Ω, and total impedance Z = 60 Ω. Calculate the active power. | 100 | W | yes |
| CH418 | CH | circuit_power | A series AC circuit has RMS voltage U = 120 V, resistance R = 40 Ω, and total impedance Z = 80 Ω. Calculate the active power. | 90 | W | yes |
| CH419 | CH | circuit_power | A series AC circuit has RMS voltage U = 150 V, resistance R = 45 Ω, and total impedance Z = 75 Ω. Calculate the active power. | 180 | W | yes |
| CH420 | CH | circuit_power | A series AC circuit has RMS voltage U = 180 V, resistance R = 50 Ω, and total impedance Z = 90 Ω. Calculate the active power. | 200 | W | yes |
| CH421 | CH | circuit_power | A series AC circuit has RMS voltage U = 200 V, resistance R = 64 Ω, and total impedance Z = 100 Ω. Calculate the active power. | 256 | W | yes |
| CH422 | CH | circuit_power | A series AC circuit has RMS voltage U = 90 V, resistance R = 24 Ω, and total impedance Z = 45 Ω. Calculate the active power. | 96 | W | yes |
| CH423 | CH | circuit_power | A series AC circuit has RMS voltage U = 160 V, resistance R = 48 Ω, and total impedance Z = 80 Ω. Calculate the active power. | 192 | W | yes |
| CH424 | CH | circuit_power | A series AC circuit has RMS voltage U = 220 V, resistance R = 55 Ω, and total impedance Z = 110 Ω. Calculate the active power. | 220 | W | yes |
| CH425 | CH | circuit_power | A series AC circuit has RMS voltage U = 130 V, resistance R = 30 Ω, and total impedance Z = 65 Ω. Calculate the active power. | 120 | W | yes |
| CH426 | CH | circuit_power | A series AC circuit has RMS voltage U = 240 V, resistance R = 72 Ω, and total impedance Z = 120 Ω. Calculate the active power. | 288 | W | yes |
| THCB192 | THCB | circuit_power | A resistor is connected to a voltage U = 12 V and carries current I = 2.5 A. Calculate the electric power. | 30 | W | yes |
| THCB193 | THCB | circuit_power | A resistor is connected to a voltage U = 24 V and carries current I = 1.25 A. Calculate the electric power. | 30 | W | yes |
| THCB194 | THCB | circuit_power | A resistor is connected to a voltage U = 9 V and carries current I = 0.8 A. Calculate the electric power. | 7.2 | W | yes |
| THCB195 | THCB | circuit_power | A resistor is connected to a voltage U = 36 V and carries current I = 0.5 A. Calculate the electric power. | 18 | W | yes |
| THCB196 | THCB | circuit_power | A resistor is connected to a voltage U = 48 V and carries current I = 1.5 A. Calculate the electric power. | 72 | W | yes |
| THCB197 | THCB | circuit_power | A resistor R = 6 Ω is connected across a voltage U = 18 V. Calculate the power dissipated. | 54 | W | yes |
| THCB198 | THCB | circuit_power | A resistor R = 15 Ω is connected across a voltage U = 30 V. Calculate the power dissipated. | 60 | W | yes |
| THCB199 | THCB | circuit_power | A resistor R = 21 Ω is connected across a voltage U = 42 V. Calculate the power dissipated. | 84 | W | yes |
| THCB200 | THCB | circuit_power | A resistor R = 20 Ω is connected across a voltage U = 60 V. Calculate the power dissipated. | 180 | W | yes |
| THCB201 | THCB | circuit_power | A resistor R = 24 Ω is connected across a voltage U = 72 V. Calculate the power dissipated. | 216 | W | yes |
| THCB202 | THCB | circuit_power | A current I = 1.5 A flows through a resistor R = 8 Ω. Calculate the Joule power. | 18 | W | yes |
| THCB203 | THCB | circuit_power | A current I = 2.0 A flows through a resistor R = 12 Ω. Calculate the Joule power. | 48 | W | yes |
| THCB204 | THCB | circuit_power | A current I = 0.75 A flows through a resistor R = 32 Ω. Calculate the Joule power. | 18 | W | yes |
| THCB205 | THCB | circuit_power | A current I = 3.0 A flows through a resistor R = 5 Ω. Calculate the Joule power. | 45 | W | yes |
| THCB206 | THCB | circuit_power | A current I = 2.5 A flows through a resistor R = 16 Ω. Calculate the Joule power. | 100 | W | yes |
| THCB207 | THCB | circuit_power | An electrical device consumes energy A = 75 J in t = 4 s. Calculate its average power. | 18.75 | W | yes |
| THCB208 | THCB | circuit_power | An electrical device consumes energy A = 120 J in t = 5 s. Calculate its average power. | 24 | W | yes |
| THCB209 | THCB | circuit_power | An electrical device consumes energy A = 360 J in t = 12 s. Calculate its average power. | 30 | W | yes |
| THCB210 | THCB | circuit_power | An electrical device consumes energy A = 540 J in t = 18 s. Calculate its average power. | 30 | W | yes |
| THCB211 | THCB | circuit_power | An electrical device consumes energy A = 900 J in t = 30 s. Calculate its average power. | 30 | W | yes |
| THCB212 | THCB | circuit_power | A resistor R = 10 Ω carries current I = 2.0 A. Calculate the rate of heat production. | 40 | W | yes |
| THCB213 | THCB | circuit_power | A resistor R = 25 Ω carries current I = 1.2 A. Calculate the rate of heat production. | 36 | W | yes |
| THCB214 | THCB | circuit_power | A resistor R = 40 Ω carries current I = 0.8 A. Calculate the rate of heat production. | 25.6 | W | yes |
| THCB215 | THCB | circuit_power | A resistor R = 6 Ω carries current I = 3.5 A. Calculate the rate of heat production. | 73.5 | W | yes |
| THCB216 | THCB | circuit_power | A resistor R = 16 Ω carries current I = 2.25 A. Calculate the rate of heat production. | 81 | W | yes |
| CH427 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 12 Ω and X_C = 108 Ω. The source frequency is increased 3 times and the RMS source voltage is U = 96 V. What is the RMS... | 96 | V | yes |
| CH428 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 18 Ω and X_C = 162 Ω. The source frequency is increased 3 times and the RMS source voltage is U = 150 V. What is the RM... | 150 | V | yes |
| CH429 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 20 Ω and X_C = 180 Ω. The source frequency is increased 3 times and the RMS source voltage is U = 220 V. What is the RM... | 220 | V | yes |
| CH430 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 7 Ω and X_C = 112 Ω. The source frequency is increased 4 times and the RMS source voltage is U = 80 V. What is the RMS ... | 80 | V | yes |
| CH431 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 10 Ω and X_C = 160 Ω. The source frequency is increased 4 times and the RMS source voltage is U = 120 V. What is the RM... | 120 | V | yes |
| CH432 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 15 Ω and X_C = 240 Ω. The source frequency is increased 4 times and the RMS source voltage is U = 180 V. What is the RM... | 180 | V | yes |
| CH433 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 8 Ω and X_C = 200 Ω. The source frequency is increased 5 times and the RMS source voltage is U = 100 V. What is the RMS... | 100 | V | yes |
| CH434 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 11 Ω and X_C = 275 Ω. The source frequency is increased 5 times and the RMS source voltage is U = 165 V. What is the RM... | 165 | V | yes |
| CH435 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 6 Ω and X_C = 216 Ω. The source frequency is increased 6 times and the RMS source voltage is U = 144 V. What is the RMS... | 144 | V | yes |
| CH436 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 9 Ω and X_C = 324 Ω. The source frequency is increased 6 times and the RMS source voltage is U = 216 V. What is the RMS... | 216 | V | yes |
| CH437 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 14 Ω and X_C = 126 Ω. The source frequency is increased 3 times and the RMS source voltage is U = 210 V. What is the RM... | 210 | V | yes |
| CH438 | CH | circuit_resistance | In a series RLC circuit, the initial reactances are X_L = 16 Ω and X_C = 256 Ω. The source frequency is increased 4 times and the RMS source voltage is U = 128 V. What is the RM... | 128 | V | yes |
| CH439 | CH | circuit_resistance | A series RLC circuit has X_L = 12 Ω, X_C = 108 Ω, and R = 30 Ω. If the frequency is increased 3 times while the RMS voltage is U = 90 V, calculate the RMS current. | 3 | A | yes |
| CH440 | CH | circuit_resistance | A series RLC circuit has X_L = 18 Ω, X_C = 162 Ω, and R = 45 Ω. If the frequency is increased 3 times while the RMS voltage is U = 135 V, calculate the RMS current. | 3 | A | yes |
| CH441 | CH | circuit_resistance | A series RLC circuit has X_L = 8 Ω, X_C = 128 Ω, and R = 25 Ω. If the frequency is increased 4 times while the RMS voltage is U = 100 V, calculate the RMS current. | 4 | A | yes |
| CH442 | CH | circuit_resistance | A series RLC circuit has X_L = 10 Ω, X_C = 160 Ω, and R = 40 Ω. If the frequency is increased 4 times while the RMS voltage is U = 160 V, calculate the RMS current. | 4 | A | yes |
| CH443 | CH | circuit_resistance | A series RLC circuit has X_L = 6 Ω, X_C = 150 Ω, and R = 20 Ω. If the frequency is increased 5 times while the RMS voltage is U = 80 V, calculate the RMS current. | 4 | A | yes |
| CH444 | CH | circuit_resistance | A series RLC circuit has X_L = 9 Ω, X_C = 225 Ω, and R = 30 Ω. If the frequency is increased 5 times while the RMS voltage is U = 120 V, calculate the RMS current. | 4 | A | yes |
| CH445 | CH | circuit_resistance | A series RLC circuit has X_L = 5 Ω, X_C = 180 Ω, and R = 24 Ω. If the frequency is increased 6 times while the RMS voltage is U = 144 V, calculate the RMS current. | 6 | A | yes |
| CH446 | CH | circuit_resistance | A series RLC circuit has X_L = 7 Ω, X_C = 252 Ω, and R = 42 Ω. If the frequency is increased 6 times while the RMS voltage is U = 210 V, calculate the RMS current. | 5 | A | yes |
| CH447 | CH | circuit_resistance | A series RLC circuit has X_L = 14 Ω, X_C = 126 Ω, and R = 70 Ω. If the frequency is increased 3 times while the RMS voltage is U = 210 V, calculate the RMS current. | 3 | A | yes |
| CH448 | CH | circuit_resistance | A series RLC circuit has X_L = 16 Ω, X_C = 256 Ω, and R = 64 Ω. If the frequency is increased 4 times while the RMS voltage is U = 192 V, calculate the RMS current. | 3 | A | yes |
| CH449 | CH | circuit_resistance | A series RLC circuit has X_L = 11 Ω, X_C = 275 Ω, and R = 55 Ω. If the frequency is increased 5 times while the RMS voltage is U = 220 V, calculate the RMS current. | 4 | A | yes |
| CH450 | CH | circuit_resistance | A series RLC circuit has X_L = 13 Ω, X_C = 468 Ω, and R = 78 Ω. If the frequency is increased 6 times while the RMS voltage is U = 234 V, calculate the RMS current. | 3 | A | yes |
| CH451 | CH | circuit_resistance | A series RLC circuit has X_L = 20 Ω, X_C = 180 Ω, and R = 50 Ω. If the frequency is increased 3 times while the RMS voltage is U = 200 V, calculate the RMS current. | 4 | A | yes |
| THCB217 | THCB | circuit_resistance | A resistor has voltage U = 12 V across it and current I = 0.6 A through it. Calculate its resistance. | 20 | Ω | yes |
| THCB218 | THCB | circuit_resistance | A resistor has voltage U = 24 V across it and current I = 1.2 A through it. Calculate its resistance. | 20 | Ω | yes |
| THCB219 | THCB | circuit_resistance | A resistor has voltage U = 9 V across it and current I = 0.3 A through it. Calculate its resistance. | 30 | Ω | yes |
| THCB220 | THCB | circuit_resistance | A resistor has voltage U = 36 V across it and current I = 1.5 A through it. Calculate its resistance. | 24 | Ω | yes |
| THCB221 | THCB | circuit_resistance | A resistor has voltage U = 48 V across it and current I = 2.0 A through it. Calculate its resistance. | 24 | Ω | yes |
| THCB222 | THCB | circuit_resistance | Two resistors R1 = 5 Ω and R2 = 7 Ω are connected in series. Calculate the equivalent resistance. | 12 | Ω | yes |
| THCB223 | THCB | circuit_resistance | Two resistors R1 = 12 Ω and R2 = 18 Ω are connected in series. Calculate the equivalent resistance. | 30 | Ω | yes |
| THCB224 | THCB | circuit_resistance | Two resistors R1 = 3.3 Ω and R2 = 6.7 Ω are connected in series. Calculate the equivalent resistance. | 10 | Ω | yes |
| THCB225 | THCB | circuit_resistance | Two resistors R1 = 15 Ω and R2 = 25 Ω are connected in series. Calculate the equivalent resistance. | 40 | Ω | yes |
| THCB226 | THCB | circuit_resistance | Two resistors R1 = 8 Ω and R2 = 22 Ω are connected in series. Calculate the equivalent resistance. | 30 | Ω | yes |
| THCB227 | THCB | circuit_resistance | Two resistors R1 = 6 Ω and R2 = 3 Ω are connected in parallel. Calculate the equivalent resistance. | 2 | Ω | yes |
| THCB228 | THCB | circuit_resistance | Two resistors R1 = 12 Ω and R2 = 4 Ω are connected in parallel. Calculate the equivalent resistance. | 3 | Ω | yes |
| THCB229 | THCB | circuit_resistance | Two resistors R1 = 10 Ω and R2 = 15 Ω are connected in parallel. Calculate the equivalent resistance. | 6 | Ω | yes |
| THCB230 | THCB | circuit_resistance | Two resistors R1 = 20 Ω and R2 = 30 Ω are connected in parallel. Calculate the equivalent resistance. | 12 | Ω | yes |
| THCB231 | THCB | circuit_resistance | Two resistors R1 = 8 Ω and R2 = 24 Ω are connected in parallel. Calculate the equivalent resistance. | 6 | Ω | yes |
| THCB232 | THCB | circuit_resistance | A wire has resistivity ρ = 1.7e-08 Ω·m, length l = 2.0 m, and cross-sectional area S = 1e-06 m². Calculate its resistance. | 0.034 | Ω | yes |
| THCB233 | THCB | circuit_resistance | A wire has resistivity ρ = 2.8e-08 Ω·m, length l = 5.0 m, and cross-sectional area S = 2e-06 m². Calculate its resistance. | 0.07 | Ω | yes |
| THCB234 | THCB | circuit_resistance | A wire has resistivity ρ = 1.1e-06 Ω·m, length l = 0.8 m, and cross-sectional area S = 4e-06 m². Calculate its resistance. | 0.22 | Ω | yes |
| THCB235 | THCB | circuit_resistance | A wire has resistivity ρ = 4e-07 Ω·m, length l = 3.0 m, and cross-sectional area S = 1.5e-06 m². Calculate its resistance. | 0.8 | Ω | yes |
| THCB236 | THCB | circuit_resistance | A wire has resistivity ρ = 2e-06 Ω·m, length l = 1.2 m, and cross-sectional area S = 3e-06 m². Calculate its resistance. | 0.8 | Ω | yes |
| THCB237 | THCB | circuit_resistance | A resistor R = 15 Ω carries current I = 0.8 A. Calculate the voltage across the resistor. | 12 | V | yes |
| THCB238 | THCB | circuit_resistance | A resistor R = 22 Ω carries current I = 1.5 A. Calculate the voltage across the resistor. | 33 | V | yes |
| THCB239 | THCB | circuit_resistance | A resistor R = 47 Ω carries current I = 0.2 A. Calculate the voltage across the resistor. | 9.4 | V | yes |
| THCB240 | THCB | circuit_resistance | A resistor R = 33 Ω carries current I = 0.6 A. Calculate the voltage across the resistor. | 19.8 | V | yes |
| THCB241 | THCB | circuit_resistance | A resistor R = 56 Ω carries current I = 0.25 A. Calculate the voltage across the resistor. | 14 | V | yes |
| THCB242 | THCB | measurement_error | A quantity is measured as x = 5.0 with absolute uncertainty Δx = 0.1. Calculate the relative error. | 2 | % | yes |
| THCB243 | THCB | measurement_error | A quantity is measured as x = 8.0 with absolute uncertainty Δx = 0.2. Calculate the relative error. | 2.5 | % | yes |
| THCB244 | THCB | measurement_error | A quantity is measured as x = 2.5 with absolute uncertainty Δx = 0.05. Calculate the relative error. | 2 | % | yes |
| THCB245 | THCB | measurement_error | A quantity is measured as x = 12.0 with absolute uncertainty Δx = 0.3. Calculate the relative error. | 2.5 | % | yes |
| THCB246 | THCB | measurement_error | A quantity is measured as x = 0.5 with absolute uncertainty Δx = 0.01. Calculate the relative error. | 2 | % | yes |
| THCB247 | THCB | measurement_error | The measured value is 9.8, while the accepted value is 10.0. Calculate the absolute error. | 0.2 | - | yes |
| THCB248 | THCB | measurement_error | The measured value is 15.3, while the accepted value is 15.0. Calculate the absolute error. | 0.3 | - | yes |
| THCB249 | THCB | measurement_error | The measured value is 0.98, while the accepted value is 1.0. Calculate the absolute error. | 0.02 | - | yes |
| THCB250 | THCB | measurement_error | The measured value is 4.7, while the accepted value is 4.5. Calculate the absolute error. | 0.2 | - | yes |
| THCB251 | THCB | measurement_error | The measured value is 101.2, while the accepted value is 100.0. Calculate the absolute error. | 1.2 | - | yes |
| THCB252 | THCB | measurement_error | A measurement is reported as x = 3.2 ± 0.05. What is the maximum possible value? | 3.25 | - | yes |
| THCB253 | THCB | measurement_error | A measurement is reported as x = 12.6 ± 0.2. What is the maximum possible value? | 12.8 | - | yes |
| THCB254 | THCB | measurement_error | A measurement is reported as x = 0.48 ± 0.01. What is the maximum possible value? | 0.49 | - | yes |
| THCB255 | THCB | measurement_error | A measurement is reported as x = 25.0 ± 0.5. What is the maximum possible value? | 25.5 | - | yes |
| THCB256 | THCB | measurement_error | A measurement is reported as x = 7.85 ± 0.03. What is the maximum possible value? | 7.88 | - | yes |
| THCB257 | THCB | measurement_error | A measurement is reported as x = 3.2 ± 0.05. What is the minimum possible value? | 3.15 | - | yes |
| THCB258 | THCB | measurement_error | A measurement is reported as x = 12.6 ± 0.2. What is the minimum possible value? | 12.4 | - | yes |
| THCB259 | THCB | measurement_error | A measurement is reported as x = 0.48 ± 0.01. What is the minimum possible value? | 0.47 | - | yes |
| THCB260 | THCB | measurement_error | A measurement is reported as x = 25.0 ± 0.5. What is the minimum possible value? | 24.5 | - | yes |
| THCB261 | THCB | measurement_error | A measurement is reported as x = 7.85 ± 0.03. What is the minimum possible value? | 7.82 | - | yes |
| THCB262 | THCB | measurement_error | Power is calculated from P = UI, with U = 12.0 ± 0.2 V and I = 0.5 ± 0.01 A. Calculate the relative error of P. | 3.667 | % | yes |
| THCB263 | THCB | measurement_error | Power is calculated from P = UI, with U = 9.0 ± 0.1 V and I = 0.3 ± 0.01 A. Calculate the relative error of P. | 4.444 | % | yes |
| THCB264 | THCB | measurement_error | Power is calculated from P = UI, with U = 24.0 ± 0.4 V and I = 1.2 ± 0.02 A. Calculate the relative error of P. | 3.333 | % | yes |
| THCB265 | THCB | measurement_error | Power is calculated from P = UI, with U = 5.0 ± 0.05 V and I = 0.25 ± 0.005 A. Calculate the relative error of P. | 3 | % | yes |
| THCB266 | THCB | measurement_error | Power is calculated from P = UI, with U = 36.0 ± 0.6 V and I = 2.0 ± 0.05 A. Calculate the relative error of P. | 4.167 | % | yes |
| THCB267 | THCB | measurement_error | Resistance is calculated by R = U/I, where U = 6.0 ± 0.1 V and I = 0.3 ± 0.01 A. Calculate the absolute error of R. | 1 | Ω | yes |
| THCB268 | THCB | measurement_error | Resistance is calculated by R = U/I, where U = 12.0 ± 0.2 V and I = 0.6 ± 0.02 A. Calculate the absolute error of R. | 1 | Ω | yes |
| THCB269 | THCB | measurement_error | Resistance is calculated by R = U/I, where U = 9.0 ± 0.15 V and I = 0.45 ± 0.015 A. Calculate the absolute error of R. | 1 | Ω | yes |
| THCB270 | THCB | measurement_error | Resistance is calculated by R = U/I, where U = 15.0 ± 0.3 V and I = 0.75 ± 0.025 A. Calculate the absolute error of R. | 1.067 | Ω | yes |
| THCB271 | THCB | measurement_error | Resistance is calculated by R = U/I, where U = 4.5 ± 0.05 V and I = 0.15 ± 0.005 A. Calculate the absolute error of R. | 1.333 | Ω | yes |
| THCB272 | THCB | measurement_error | A quantity y = a + b is computed from a = 3.2 ± 0.1 and b = 4.6 ± 0.2. Calculate the absolute error of y. | 0.3 | - | yes |
| THCB273 | THCB | measurement_error | A quantity y = a + b is computed from a = 12.5 ± 0.3 and b = 8.0 ± 0.2. Calculate the absolute error of y. | 0.5 | - | yes |
| THCB274 | THCB | measurement_error | A quantity y = a + b is computed from a = 0.75 ± 0.02 and b = 1.25 ± 0.03. Calculate the absolute error of y. | 0.05 | - | yes |
| THCB275 | THCB | measurement_error | A quantity y = a + b is computed from a = 25.0 ± 0.5 and b = 15.0 ± 0.4. Calculate the absolute error of y. | 0.9 | - | yes |
| THCB276 | THCB | measurement_error | A quantity y = a + b is computed from a = 6.8 ± 0.1 and b = 2.4 ± 0.05. Calculate the absolute error of y. | 0.15 | - | yes |
| THCB277 | THCB | measurement_error | A quantity y = ab is computed from a = 4.0 ± 0.1 and b = 2.0 ± 0.05. Calculate the absolute error of y. | 0.4 | - | yes |
| THCB278 | THCB | measurement_error | A quantity y = ab is computed from a = 6.0 ± 0.2 and b = 3.0 ± 0.1. Calculate the absolute error of y. | 1.2 | - | yes |
| THCB279 | THCB | measurement_error | A quantity y = ab is computed from a = 2.5 ± 0.05 and b = 8.0 ± 0.2. Calculate the absolute error of y. | 0.9 | - | yes |
| THCB280 | THCB | measurement_error | A quantity y = ab is computed from a = 10.0 ± 0.3 and b = 1.5 ± 0.03. Calculate the absolute error of y. | 0.75 | - | yes |
| THCB281 | THCB | measurement_error | A quantity y = ab is computed from a = 12.0 ± 0.4 and b = 2.5 ± 0.05. Calculate the absolute error of y. | 1.6 | - | yes |
| THCB282 | THCB | measurement_error | Three repeated measurements give 9.8, 10.0, and 10.2. Calculate the average measured value. | 10 | - | yes |
| THCB283 | THCB | measurement_error | Three repeated measurements give 4.9, 5.1, and 5.0. Calculate the average measured value. | 5 | - | yes |
| THCB284 | THCB | measurement_error | Three repeated measurements give 0.48, 0.51, and 0.5. Calculate the average measured value. | 0.497 | - | yes |
| THCB285 | THCB | measurement_error | Three repeated measurements give 19.8, 20.1, and 20.0. Calculate the average measured value. | 19.967 | - | yes |
| THCB286 | THCB | measurement_error | Three repeated measurements give 2.95, 3.05, and 3.0. Calculate the average measured value. | 3 | - | yes |
| DDT444 | DDT | LC_oscillation | An ideal LC circuit has inductance L = 0.2 H and capacitance C = 50 μF. Calculate the angular frequency of oscillation. | 316.228 | rad/s | yes |
| DDT445 | DDT | LC_oscillation | An ideal LC circuit has inductance L = 0.1 H and capacitance C = 100 μF. Calculate the angular frequency of oscillation. | 316.228 | rad/s | yes |
| DDT446 | DDT | LC_oscillation | An ideal LC circuit has inductance L = 0.5 H and capacitance C = 20 μF. Calculate the angular frequency of oscillation. | 316.228 | rad/s | yes |
| DDT447 | DDT | LC_oscillation | An ideal LC circuit has inductance L = 0.25 H and capacitance C = 40 μF. Calculate the angular frequency of oscillation. | 316.228 | rad/s | yes |
| DDT448 | DDT | LC_oscillation | An ideal LC circuit has inductance L = 0.4 H and capacitance C = 25 μF. Calculate the angular frequency of oscillation. | 316.228 | rad/s | yes |
| DDT449 | DDT | LC_oscillation | An ideal LC circuit has inductance L = 0.08 H and capacitance C = 125 μF. Calculate the angular frequency of oscillation. | 316.228 | rad/s | yes |
| DDT450 | DDT | LC_oscillation | An ideal LC circuit has inductance L = 0.16 H and capacitance C = 62.5 μF. Calculate the angular frequency of oscillation. | 316.228 | rad/s | yes |
| DDT451 | DDT | LC_oscillation | An ideal LC circuit has inductance L = 0.32 H and capacitance C = 31.25 μF. Calculate the angular frequency of oscillation. | 316.228 | rad/s | yes |
| DDT452 | DDT | LC_oscillation | An ideal LC circuit has inductance L = 0.05 H and capacitance C = 200 μF. Calculate the angular frequency of oscillation. | 316.228 | rad/s | yes |
| DDT453 | DDT | LC_oscillation | An ideal LC circuit has inductance L = 0.125 H and capacitance C = 80 μF. Calculate the angular frequency of oscillation. | 316.228 | rad/s | yes |
| DDT454 | DDT | LC_oscillation | An LC oscillator has L = 0.2 H and C = 50 μF. Calculate the oscillation frequency. | 50.329 | Hz | yes |
| DDT455 | DDT | LC_oscillation | An LC oscillator has L = 0.1 H and C = 100 μF. Calculate the oscillation frequency. | 50.329 | Hz | yes |
| DDT456 | DDT | LC_oscillation | An LC oscillator has L = 0.5 H and C = 20 μF. Calculate the oscillation frequency. | 50.329 | Hz | yes |
| DDT457 | DDT | LC_oscillation | An LC oscillator has L = 0.25 H and C = 40 μF. Calculate the oscillation frequency. | 50.329 | Hz | yes |
| DDT458 | DDT | LC_oscillation | An LC oscillator has L = 0.4 H and C = 25 μF. Calculate the oscillation frequency. | 50.329 | Hz | yes |
| DDT459 | DDT | LC_oscillation | An LC oscillator has L = 0.08 H and C = 125 μF. Calculate the oscillation frequency. | 50.329 | Hz | yes |
| DDT460 | DDT | LC_oscillation | An LC oscillator has L = 0.16 H and C = 62.5 μF. Calculate the oscillation frequency. | 50.329 | Hz | yes |
| DDT461 | DDT | LC_oscillation | An LC oscillator has L = 0.32 H and C = 31.25 μF. Calculate the oscillation frequency. | 50.329 | Hz | yes |
| DDT462 | DDT | LC_oscillation | An LC oscillator has L = 0.05 H and C = 200 μF. Calculate the oscillation frequency. | 50.329 | Hz | yes |
| DDT463 | DDT | LC_oscillation | An LC oscillator has L = 0.125 H and C = 80 μF. Calculate the oscillation frequency. | 50.329 | Hz | yes |
| DDT464 | DDT | LC_oscillation | An ideal LC circuit has L = 0.2 H and C = 50 μF. Calculate the oscillation period. | 0.019869 | s | yes |
| DDT465 | DDT | LC_oscillation | An ideal LC circuit has L = 0.1 H and C = 100 μF. Calculate the oscillation period. | 0.019869 | s | yes |
| DDT466 | DDT | LC_oscillation | An ideal LC circuit has L = 0.5 H and C = 20 μF. Calculate the oscillation period. | 0.019869 | s | yes |
| DDT467 | DDT | LC_oscillation | An ideal LC circuit has L = 0.25 H and C = 40 μF. Calculate the oscillation period. | 0.019869 | s | yes |
| DDT468 | DDT | LC_oscillation | An ideal LC circuit has L = 0.4 H and C = 25 μF. Calculate the oscillation period. | 0.019869 | s | yes |
| DDT469 | DDT | LC_oscillation | An ideal LC circuit has L = 0.08 H and C = 125 μF. Calculate the oscillation period. | 0.019869 | s | yes |
| DDT470 | DDT | LC_oscillation | An ideal LC circuit has L = 0.16 H and C = 62.5 μF. Calculate the oscillation period. | 0.019869 | s | yes |
| DDT471 | DDT | LC_oscillation | An ideal LC circuit has L = 0.32 H and C = 31.25 μF. Calculate the oscillation period. | 0.019869 | s | yes |
| DDT472 | DDT | LC_oscillation | An ideal LC circuit has L = 0.05 H and C = 200 μF. Calculate the oscillation period. | 0.019869 | s | yes |
| DDT473 | DDT | LC_oscillation | An ideal LC circuit has L = 0.125 H and C = 80 μF. Calculate the oscillation period. | 0.019869 | s | yes |
| DDT474 | DDT | LC_oscillation | In an LC oscillator, the capacitor has C = 20 μF and maximum voltage U0 = 12 V. Calculate the total electromagnetic energy. | 0.00144 | J | yes |
| DDT475 | DDT | LC_oscillation | In an LC oscillator, the capacitor has C = 50 μF and maximum voltage U0 = 10 V. Calculate the total electromagnetic energy. | 0.0025 | J | yes |
| DDT476 | DDT | LC_oscillation | In an LC oscillator, the capacitor has C = 100 μF and maximum voltage U0 = 6 V. Calculate the total electromagnetic energy. | 0.0018 | J | yes |
| DDT477 | DDT | LC_oscillation | In an LC oscillator, the capacitor has C = 25 μF and maximum voltage U0 = 20 V. Calculate the total electromagnetic energy. | 0.005 | J | yes |
| DDT478 | DDT | LC_oscillation | In an LC oscillator, the capacitor has C = 80 μF and maximum voltage U0 = 15 V. Calculate the total electromagnetic energy. | 0.009 | J | yes |
| DDT479 | DDT | LC_oscillation | In an LC oscillator, C = 20 μF and the maximum capacitor voltage is U0 = 12 V. Calculate the maximum charge. | 0.00024 | C | yes |
| DDT480 | DDT | LC_oscillation | In an LC oscillator, C = 50 μF and the maximum capacitor voltage is U0 = 10 V. Calculate the maximum charge. | 0.0005 | C | yes |
| DDT481 | DDT | LC_oscillation | In an LC oscillator, C = 100 μF and the maximum capacitor voltage is U0 = 6 V. Calculate the maximum charge. | 0.0006 | C | yes |
| DDT482 | DDT | LC_oscillation | In an LC oscillator, C = 25 μF and the maximum capacitor voltage is U0 = 20 V. Calculate the maximum charge. | 0.0005 | C | yes |
| DDT483 | DDT | LC_oscillation | In an LC oscillator, C = 80 μF and the maximum capacitor voltage is U0 = 15 V. Calculate the maximum charge. | 0.0012 | C | yes |
| DDT484 | DDT | LC_oscillation | An LC oscillator has L = 0.2 H, C = 50 μF, and maximum capacitor voltage U0 = 12 V. Calculate the maximum current. | 0.189737 | A | yes |
| DDT485 | DDT | LC_oscillation | An LC oscillator has L = 0.1 H, C = 100 μF, and maximum capacitor voltage U0 = 10 V. Calculate the maximum current. | 0.316228 | A | yes |
| DDT486 | DDT | LC_oscillation | An LC oscillator has L = 0.5 H, C = 20 μF, and maximum capacitor voltage U0 = 6 V. Calculate the maximum current. | 0.037947 | A | yes |
| DDT487 | DDT | LC_oscillation | An LC oscillator has L = 0.25 H, C = 40 μF, and maximum capacitor voltage U0 = 20 V. Calculate the maximum current. | 0.252982 | A | yes |
| DDT488 | DDT | LC_oscillation | An LC oscillator has L = 0.4 H, C = 25 μF, and maximum capacitor voltage U0 = 15 V. Calculate the maximum current. | 0.118585 | A | yes |

## Adapter Dataset Recommendation
- Keep `verified_golden_official_safe.csv` as the provenance and compliance reference.
- Generate synthetic explanations with a disclosed open-source model of at most 8B parameters.
- Audit accepted and rejected synthetic explanations before adapter training.
