# Leave-one-channel-out sensitivity (D1)

**POST HOC, EXPLORATORY: leave-one-channel-out sensitivity added on 2026-10-22 after an external review; not pre-registered; the frozen m2a-freeze results and verdicts are unchanged.**

Every confirmatory channel is left out in turn (17 fits per effect and model). Channel-weighted = each channel equal, as frozen; switch-weighted = each switch point equal. p = exact one-sided sign-flip over the remaining 16 channels, unadjusted. 'Direction changes' lists left-out channels whose fit is on the other side of the null from the full-sample estimate; 'SESOI crossings' those whose fit is on the other side of the smallest effect of interest. Channel IDs are the dataset's speaker_id.

Reproduction check (Part A): every full-sample estimate and sign-flip p recomputed from the frozen per-channel statistics equals the frozen value (max |diff| 5.6e-17).

### Part A: exact, from the frozen per-channel statistics

| Effect | Model | Weighting | Full | Leave-one-out min (without) | max (without) | Direction changes | SESOI crossings | max p (without) |
|---|---|---|---|---|---|---|---|---|
| h1_dz | E1 | channel | 0.271 | 0.254 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.290 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | none | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h1_dz | E1 | switch | 0.250 | 0.232 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.267 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | none |  |
| h1_dz | E2 | channel | 0.194 | 0.181 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.204 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | UCIKslGb0pGieZTCn8341Xsw, UCJinKAyRTpHJOPE0YgCMhog, UChx9-bynjfm6Js3k54Sf1Mg, UCiVc0CAFFWBOd2Zf3dQwThw, UCs_i-XIuSGGLQu-OHrWeTCA | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h1_dz | E2 | switch | 0.193 | 0.180 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.202 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | UCAh6gaPtNIBBW_qNss4flcA, UCiVc0CAFFWBOd2Zf3dQwThw, UCvc1Xg1qWSfQptoCEq13pEA |  |
| h1_dz | E3 | channel | 0.254 | 0.245 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.266 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | none | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h1_dz | E3 | switch | 0.249 | 0.234 (UCJinKAyRTpHJOPE0YgCMhog) | 0.263 (UCAh6gaPtNIBBW_qNss4flcA) | none | none |  |
| h2p0_auc | E1 | channel | 0.584 | 0.578 (UChx9-bynjfm6Js3k54Sf1Mg) | 0.589 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | none | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h2p0_auc | E1 | switch | 0.578 | 0.572 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.582 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | none |  |
| h2p0_auc | E2 | channel | 0.573 | 0.567 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.577 (UCvc1Xg1qWSfQptoCEq13pEA) | none | none | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h2p0_auc | E2 | switch | 0.569 | 0.563 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.576 (UCvc1Xg1qWSfQptoCEq13pEA) | none | none |  |
| h2p0_auc | E3 | channel | 0.582 | 0.579 (UCyPr9KIODQt5J1oOGbz44Hg) | 0.586 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | none | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h2p0_auc | E3 | switch | 0.580 | 0.576 (UCyPr9KIODQt5J1oOGbz44Hg) | 0.585 (UCAh6gaPtNIBBW_qNss4flcA) | none | none |  |
| dfar_p0L1 | E1 | channel | +2.46 pp | +1.17 pp (UChx9-bynjfm6Js3k54Sf1Mg) | +2.92 pp (UCiVc0CAFFWBOd2Zf3dQwThw) | none | n/a | 0.092 (UChx9-bynjfm6Js3k54Sf1Mg) |
| dfar_p0L1 | E1 | switch | +0.78 pp | +0.48 pp (UCZfM_5lOt0Rfs9lY0kZ5glw) | +1.26 pp (UCJinKAyRTpHJOPE0YgCMhog) | none | n/a |  |
| dfar_p0L1 | E2 | channel | +2.95 pp | +2.19 pp (UCA4izoAnn46iBh3TupN3ADQ) | +3.63 pp (UCIKslGb0pGieZTCn8341Xsw) | none | n/a | 0.045 (UCA4izoAnn46iBh3TupN3ADQ) |
| dfar_p0L1 | E2 | switch | +2.00 pp | +1.61 pp (UCZfM_5lOt0Rfs9lY0kZ5glw) | +2.66 pp (UCAh6gaPtNIBBW_qNss4flcA) | none | n/a |  |
| dfar_p0L1 | E3 | channel | +2.73 pp | +2.03 pp (UCA4izoAnn46iBh3TupN3ADQ) | +2.99 pp (UCAh6gaPtNIBBW_qNss4flcA) | none | n/a | 0.0067 (UCA4izoAnn46iBh3TupN3ADQ) |
| dfar_p0L1 | E3 | switch | +1.17 pp | +0.97 pp (UCyPr9KIODQt5J1oOGbz44Hg) | +1.63 pp (UCAh6gaPtNIBBW_qNss4flcA) | none | n/a |  |
| h3_shift_sd | E1 | channel | -0.215 | -0.224 (UCJinKAyRTpHJOPE0YgCMhog) | -0.212 (UCZfM_5lOt0Rfs9lY0kZ5glw) | none | none | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h3_shift_sd | E2 | channel | -0.173 | -0.185 (UCJinKAyRTpHJOPE0YgCMhog) | -0.167 (UCy-Q66KQYSQAUVKvdNxgKbw) | none | none | 3.1e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h3_shift_sd | E3 | channel | -0.203 | -0.213 (UCA4izoAnn46iBh3TupN3ADQ) | -0.196 (UCvc1Xg1qWSfQptoCEq13pEA) | none | UCZfM_5lOt0Rfs9lY0kZ5glw, UChx9-bynjfm6Js3k54Sf1Mg, UCjD3bnXfPtoGVfZ4PmY9sBQ, UCvc1Xg1qWSfQptoCEq13pEA | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |

Part B, rebuilt inputs: full-sample difference from the frozen value

| Model | h2p0_auc | dfar_p0L1 | det_auc | det_w1_auc |
|---|---|---|---|---|
| E1 | 0.584 vs 0.584 | +2.51 pp vs +2.46 pp | 0.488 vs 0.488 | 0.480 vs 0.480 |
| E2 | 0.573 vs 0.573 | +3.10 pp vs +2.95 pp | 0.485 vs 0.485 | 0.477 vs 0.477 |
| E3 | 0.582 vs 0.582 | +2.73 pp vs +2.73 pp | 0.493 vs 0.493 | 0.486 vs 0.486 |

### Part B: rebuilt inputs

| Effect | Model | Weighting | Full | Leave-one-out min (without) | max (without) | Direction changes | SESOI crossings | max p (without) |
|---|---|---|---|---|---|---|---|---|
| h2p0_auc | E1 | channel | 0.584 | 0.578 (UChx9-bynjfm6Js3k54Sf1Mg) | 0.589 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | none | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h2p0_auc | E1 | switch | 0.578 | 0.572 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.582 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | none |  |
| h2p0_auc | E2 | channel | 0.573 | 0.567 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.577 (UCvc1Xg1qWSfQptoCEq13pEA) | none | none | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h2p0_auc | E2 | switch | 0.569 | 0.563 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.576 (UCvc1Xg1qWSfQptoCEq13pEA) | none | none |  |
| h2p0_auc | E3 | channel | 0.582 | 0.579 (UCyPr9KIODQt5J1oOGbz44Hg) | 0.586 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | none | 1.5e-05 (UC1a2Vvzr7NJMtkIoV9c6Opw) |
| h2p0_auc | E3 | switch | 0.580 | 0.576 (UCyPr9KIODQt5J1oOGbz44Hg) | 0.585 (UCAh6gaPtNIBBW_qNss4flcA) | none | none |  |
| dfar_p0L1 | E1 | channel | +2.51 pp | +1.22 pp (UChx9-bynjfm6Js3k54Sf1Mg) | +2.97 pp (UCiVc0CAFFWBOd2Zf3dQwThw) | none | n/a | 0.084 (UChx9-bynjfm6Js3k54Sf1Mg) |
| dfar_p0L1 | E1 | switch | +0.81 pp | +0.52 pp (UCZfM_5lOt0Rfs9lY0kZ5glw) | +1.29 pp (UCAh6gaPtNIBBW_qNss4flcA) | none | n/a |  |
| dfar_p0L1 | E2 | channel | +3.10 pp | +2.18 pp (UCA4izoAnn46iBh3TupN3ADQ) | +3.79 pp (UCIKslGb0pGieZTCn8341Xsw) | none | n/a | 0.046 (UCA4izoAnn46iBh3TupN3ADQ) |
| dfar_p0L1 | E2 | switch | +2.01 pp | +1.61 pp (UCZfM_5lOt0Rfs9lY0kZ5glw) | +2.68 pp (UCAh6gaPtNIBBW_qNss4flcA) | none | n/a |  |
| dfar_p0L1 | E3 | channel | +2.73 pp | +2.03 pp (UCA4izoAnn46iBh3TupN3ADQ) | +2.99 pp (UCAh6gaPtNIBBW_qNss4flcA) | none | n/a | 0.0067 (UCA4izoAnn46iBh3TupN3ADQ) |
| dfar_p0L1 | E3 | switch | +1.17 pp | +0.97 pp (UCyPr9KIODQt5J1oOGbz44Hg) | +1.63 pp (UCAh6gaPtNIBBW_qNss4flcA) | none | n/a |  |
| det_auc | E1 | channel | 0.488 | 0.485 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.491 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a | 0.99 (UCUj0DJEMJRzBgrFOnDqOZig) |
| det_auc | E1 | switch | 0.489 | 0.485 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.492 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a |  |
| det_auc | E2 | channel | 0.485 | 0.482 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.487 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a | 1 (UCUj0DJEMJRzBgrFOnDqOZig) |
| det_auc | E2 | switch | 0.486 | 0.482 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.489 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a |  |
| det_auc | E3 | channel | 0.493 | 0.491 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.495 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a | 0.95 (UCUj0DJEMJRzBgrFOnDqOZig) |
| det_auc | E3 | switch | 0.495 | 0.493 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.498 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a |  |
| det_w1_auc | E1 | channel | 0.480 | 0.475 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.484 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a | 0.99 (UCUj0DJEMJRzBgrFOnDqOZig) |
| det_w1_auc | E1 | switch | 0.483 | 0.476 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.487 (UCiVc0CAFFWBOd2Zf3dQwThw) | none | n/a |  |
| det_w1_auc | E2 | channel | 0.477 | 0.473 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.482 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a | 1 (UCUj0DJEMJRzBgrFOnDqOZig) |
| det_w1_auc | E2 | switch | 0.480 | 0.473 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.484 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a |  |
| det_w1_auc | E3 | channel | 0.486 | 0.483 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.491 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a | 0.96 (UCUj0DJEMJRzBgrFOnDqOZig) |
| det_w1_auc | E3 | switch | 0.489 | 0.486 (UCUj0DJEMJRzBgrFOnDqOZig) | 0.495 (UCIKslGb0pGieZTCn8341Xsw) | none | n/a |  |

## Possible shared speaker (pilot and confirmatory channel)

Pilot channel `UCeq9rTo9W93NfJNP9n1eRGw`, video `zJYS-XNDP2Q`; confirmatory channel `UCOC6C8_U9PsAs3tQodDDmGw`, videos `4qYik5d_rLw`, `8ywUuBnUg0g`, `CCSTJTzx1YA`, `DxI6929EuWg`.

- `4qYik5d_rLw` vs `zJYS-XNDP2Q`: cos TitaNet 0.840, CAM++ 0.874
- `CCSTJTzx1YA` vs `zJYS-XNDP2Q`: cos TitaNet 0.838, CAM++ 0.846
- `8ywUuBnUg0g` vs `zJYS-XNDP2Q`: cos TitaNet 0.831, CAM++ 0.878
- `DxI6929EuWg` vs `zJYS-XNDP2Q`: cos TitaNet 0.815, CAM++ 0.835
- The pilot video's best cosine to its own channel: TitaNet 0.536, CAM++ 0.604 (split off by the audit).

Without the confirmatory channel (Part A, exact; channel-weighted):

| Model | H1 d_z | H2-P0 AUC | Delta FAR | H3 shift |
|---|---|---|---|---|
| E1 | 0.269 | 0.585 | +2.64 pp | -0.219 |
| E2 | 0.189 | 0.573 | +2.94 pp | -0.176 |
| E3 | 0.256 | 0.584 | +2.66 pp | -0.207 |

Without only the four matching videos (Part B, rebuilt inputs; channel-weighted):

| Model | H2-P0 AUC | Delta FAR | det AUC | det_w1 AUC |
|---|---|---|---|---|
| E1 | 0.585 | +2.69 pp | 0.489 | 0.481 |
| E2 | 0.573 | +3.09 pp | 0.486 | 0.478 |
| E3 | 0.584 | +2.66 pp | 0.494 | 0.488 |
