# M2 lambda(t) pre-tuning check v2 (validation only)

| Family | Comparison | AUC | n |
|---|---|---|---|
| det | all | 0.576 | 16207 |
| det | stratum:lt0.5 | 0.562 | 12552 |
| det | stratum:0.5to1 | 0.610 | 3165 |
| det | stratum:ge1 | 0.715 | 490 |
| det | pause:none | 0.574 | 14254 |
| det | pause:lt150 | 0.564 | 752 |
| det | pause:150to300 | 0.611 | 843 |
| det | pause:gt300 | 0.579 | 358 |
| det | phrase_vs_other_controls | 0.560 | 1339 |
| emb1 | all | 0.720 | 443 |
| emb1 | stratum:lt0.5 | nan | 0 |
| emb1 | stratum:0.5to1 | nan | 0 |
| emb1 | stratum:ge1 | 0.720 | 443 |
| emb1 | pause:none | 0.722 | 347 |
| emb1 | pause:lt150 | 0.669 | 27 |
| emb1 | pause:150to300 | 0.727 | 41 |
| emb1 | pause:gt300 | 0.732 | 28 |
| emb1 | phrase_vs_other_controls | 0.579 | 392 |
