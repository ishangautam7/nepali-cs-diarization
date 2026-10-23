# M2 lambda(t) pre-tuning check v1 (validation only)

| Family | Comparison | AUC | n |
|---|---|---|---|
| det | all | 0.580 | 17868 |
| det | stratum:lt0.5 | 0.568 | 13978 |
| det | stratum:0.5to1 | 0.612 | 3356 |
| det | stratum:ge1 | 0.714 | 534 |
| det | pause:none | 0.579 | 15654 |
| det | pause:lt150 | 0.569 | 805 |
| det | pause:150to300 | 0.602 | 939 |
| det | pause:gt300 | 0.590 | 470 |
| det | phrase_vs_other_controls | 0.572 | 1440 |
| emb1 | all | 0.723 | 436 |
| emb1 | stratum:lt0.5 | nan | 0 |
| emb1 | stratum:0.5to1 | nan | 0 |
| emb1 | stratum:ge1 | 0.723 | 436 |
| emb1 | pause:none | 0.725 | 340 |
| emb1 | pause:lt150 | 0.683 | 26 |
| emb1 | pause:150to300 | 0.722 | 43 |
| emb1 | pause:gt300 | 0.737 | 27 |
| emb1 | phrase_vs_other_controls | 0.593 | 420 |
