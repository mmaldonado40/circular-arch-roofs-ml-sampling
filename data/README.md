# Data dictionary

The master All_dataset.csv is the accepted numerical authority: 221 unique geometries, five inputs and 315 dimensionless pressure coefficients per case. It contains 201 development cases (G_* identifiers) and 20 retained test cases T_1–T_20. Order is the retained case order; design_type records LHS, CCD or 2k.

| Input | Meaning | Unit | Declared range |
|---|---|---|---|
| R/D | Roof rise / span | dimensionless | 0.10–0.50 |
| B/D | Longitudinal roof length / span | dimensionless | 1.00–3.00 |
| D | Span | m | 10–100 |
| h | Eave height | m | 5–25 |
| Angle | Wind direction: 0° normal to ridge, 90° parallel | degrees | 0–90 |

Here R is roof rise, not curvature radius. The roof/measurement/domain schematics are F1–F3 in ../results/figures/.

Cp_001–Cp_315 follow pointz{z}_{j}, with z=1…15 longitudinal measurement planes and j=1…21 points per plane: column index = (z−1)×21+j. Local analyses use planes 1, 8 and 15. Position on the roof arc is distinct from the wind-direction input Angle.

designs/ contains the nine accepted principal designs and nine additional convergence subsets. Clave equals master ID, Angulo equals Angle and v1…v315 equal Cp_001…Cp_315. Tipo records the design label. The verifier compares these fields with the master, including ordered membership. FFD denotes the retained 16-case half factorial; 2k the 32-case full factorial.

| Family | Base / +FFD / +2k cases |
|---|---|
| CCD | 43 / 59 / 75 |
| Retained reduced LHS (R-LHS) | 43 / 59 / 75 |
| Full LHS | 126 / 142 / 158 |

Convergence sizes are 13, 25, 38, 50, 63, 76, 88, 101, 113 and 126; LHS126.csv is also the N126 convergence matrix. The subsets are not all nested. The stored IDs and realized values, rather than newly generated designs, define this study.

doe_input_quantization.xlsx retains construction provenance. Its 221 “Constructed geometry” cases agree with the accepted master. Only Excel’s stored absolute-directory metadata was removed from this copy; all worksheet content, formulas and styles are unchanged. The “Mapped ML input” block has 312 alternative ratio differences from dimension-based quotients; these values must not replace the master inputs.

The master supports ML reanalysis on processed CFD outputs. Recreating the simulation campaign and the prior study’s experimental CFD validation are outside this repository’s scope. Original datasets and figures are licensed under CC BY 4.0; see ../LICENSE-DATA.md for scope and exclusions.
